/*
 * Author : Jiayuan Mao, Tete Xiao
 * Email  : maojiayuan@gmail.com, jasonhsiao97@gmail.com 
 * Date   : 07/13/2018
 * 
 * Distributed under terms of the MIT license.
 * Copyright (c) 2017 Megvii Technology Limited.
 */
/*
 * Modified By RuiminChen
 */
#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/psprroi_pooling_layer.hpp"
#include "caffe/util/gpu_util.cuh"

using std::max;
using std::min;

namespace caffe {

  template <typename Dtype>
       __device__ static Dtype PrRoIPoolingGetData(const Dtype* data, const int h, const int w, const int height, const int width)
    {
        bool overflow = (h < 0) || (w < 0) || (h >= height) || (w >= width);
        Dtype retVal = overflow ? 0.0f : data[h * width + w];
        return retVal;
    }
    
    
  template <typename Dtype>
      __device__ static Dtype PrRoIPoolingGetCoeff(Dtype dh, Dtype dw){
        dw = dw > 0 ? dw : -dw;
        dh = dh > 0 ? dh : -dh;
        return (1.0f - dh) * (1.0f - dw);
    }
    
    
  template <typename Dtype>
      __device__ static Dtype PrRoIPoolingSingleCoorIntegral(Dtype s, Dtype t, Dtype c1, Dtype c2) {
        return 0.5 * (t * t - s * s) * c2 + (t - 0.5 * t * t - s + 0.5 * s * s) * c1;
    }
  
  template <typename Dtype>
      __device__ static Dtype PrRoIPoolingInterpolation(const Dtype* data, const Dtype h, const Dtype w, const int height, const int width){
        Dtype retVal = 0.0f;
        int h1 = floorf(h);
        int w1 = floorf(w);
        retVal += PrRoIPoolingGetData(data, h1, w1, height, width) * PrRoIPoolingGetCoeff(h - Dtype(h1), w - Dtype(w1));
        h1 = floorf(h)+1;
        w1 = floorf(w);
        retVal += PrRoIPoolingGetData(data, h1, w1, height, width) * PrRoIPoolingGetCoeff(h - Dtype(h1), w - Dtype(w1));
        h1 = floorf(h);
        w1 = floorf(w)+1;
        retVal += PrRoIPoolingGetData(data, h1, w1, height, width) * PrRoIPoolingGetCoeff(h - Dtype(h1), w - Dtype(w1));
        h1 = floorf(h)+1;
        w1 = floorf(w)+1;
        retVal += PrRoIPoolingGetData(data, h1, w1, height, width) * PrRoIPoolingGetCoeff(h - Dtype(h1), w - Dtype(w1));
        return retVal;
    }
  
  template <typename Dtype>
      __device__ static Dtype PrRoIPoolingMatCalculation(const Dtype* this_data, const int s_h, const int s_w, const int e_h, const int e_w,
            const Dtype y0, const Dtype x0, const Dtype y1, const Dtype x1, const int h0, const int w0)
    {
        Dtype alpha, beta, lim_alpha, lim_beta, tmp;
        Dtype sum_out = 0;

        alpha = x0 - Dtype(s_w);
        beta = y0 - Dtype(s_h);
        lim_alpha = x1 - Dtype(s_w);
        lim_beta = y1 - Dtype(s_h);
        tmp = (lim_alpha - 0.5f * lim_alpha * lim_alpha - alpha + 0.5f * alpha * alpha) 
            * (lim_beta - 0.5f * lim_beta * lim_beta - beta + 0.5f * beta * beta);
        sum_out += PrRoIPoolingGetData(this_data, s_h, s_w, h0, w0) * tmp;

        alpha = Dtype(e_w) - x1;
        lim_alpha = Dtype(e_w) - x0;
        tmp = (lim_alpha - 0.5f * lim_alpha * lim_alpha - alpha + 0.5f * alpha * alpha) 
            * (lim_beta - 0.5f * lim_beta * lim_beta - beta + 0.5f * beta * beta);
        sum_out += PrRoIPoolingGetData(this_data, s_h, e_w, h0, w0) * tmp;

        alpha = x0 - Dtype(s_w);
        beta = Dtype(e_h) - y1;
        lim_alpha = x1 - Dtype(s_w);
        lim_beta = Dtype(e_h) - y0;
        tmp = (lim_alpha - 0.5f * lim_alpha * lim_alpha - alpha + 0.5f * alpha * alpha) 
            * (lim_beta - 0.5f * lim_beta * lim_beta - beta + 0.5f * beta * beta);
        sum_out += PrRoIPoolingGetData(this_data, e_h, s_w, h0, w0) * tmp;

        alpha = Dtype(e_w) - x1;
        lim_alpha = Dtype(e_w) - x0;
        tmp = (lim_alpha - 0.5f * lim_alpha * lim_alpha - alpha + 0.5f * alpha * alpha) 
            * (lim_beta - 0.5f * lim_beta * lim_beta - beta + 0.5f * beta * beta);   
        sum_out += PrRoIPoolingGetData(this_data, e_h, e_w, h0, w0) * tmp;

        return sum_out;
    }
  
  template <typename Dtype>
      __device__ static void PrRoIPoolingDistributeDiff(Dtype* diff, const Dtype top_diff, const int h, const int w, const int height, const int width, const Dtype coeff)
    {
        bool overflow = (h < 0) || (w < 0) || (h >= height) || (w >= width);
        if (!overflow) 
            caffe_gpu_atomic_add(top_diff * coeff, diff + h * width + w);
    }
  
  
  template <typename Dtype>
      __device__ static void PrRoIPoolingMatDistributeDiff(Dtype* diff, const Dtype top_diff, const int s_h, const int s_w, const int e_h, const int e_w,
            const Dtype y0, const Dtype x0, const Dtype y1, const Dtype x1, const int h0, const int w0)
    {
        Dtype alpha, beta, lim_alpha, lim_beta, tmp;

        alpha = x0 - Dtype(s_w);
        beta = y0 - Dtype(s_h);
        lim_alpha = x1 - Dtype(s_w);
        lim_beta = y1 - Dtype(s_h);
        tmp = (lim_alpha - 0.5f * lim_alpha * lim_alpha - alpha + 0.5f * alpha * alpha) 
            * (lim_beta - 0.5f * lim_beta * lim_beta - beta + 0.5f * beta * beta);
        PrRoIPoolingDistributeDiff(diff, top_diff, s_h, s_w, h0, w0, tmp);

        alpha = Dtype(e_w) - x1;
        lim_alpha = Dtype(e_w) - x0;
        tmp = (lim_alpha - 0.5f * lim_alpha * lim_alpha - alpha + 0.5f * alpha * alpha) 
            * (lim_beta - 0.5f * lim_beta * lim_beta - beta + 0.5f * beta * beta);
        PrRoIPoolingDistributeDiff(diff, top_diff, s_h, e_w, h0, w0, tmp);

        alpha = x0 - Dtype(s_w);
        beta = Dtype(e_h) - y1;
        lim_alpha = x1 - Dtype(s_w);
        lim_beta = Dtype(e_h) - y0;
        tmp = (lim_alpha - 0.5f * lim_alpha * lim_alpha - alpha + 0.5f * alpha * alpha) 
            * (lim_beta - 0.5f * lim_beta * lim_beta - beta + 0.5f * beta * beta);
        PrRoIPoolingDistributeDiff(diff, top_diff, e_h, s_w, h0, w0, tmp);

        alpha = Dtype(e_w) - x1;
        lim_alpha = Dtype(e_w) - x0;
        tmp = (lim_alpha - 0.5f * lim_alpha * lim_alpha - alpha + 0.5f * alpha * alpha) 
            * (lim_beta - 0.5f * lim_beta * lim_beta - beta + 0.5f * beta * beta);   
        PrRoIPoolingDistributeDiff(diff, top_diff, e_h, e_w, h0, w0, tmp);
    }
  
  
  template <typename Dtype>
  __global__ void PSPRROIPoolingForward(
    const int nthreads,
    const Dtype* bottom_data,
    const Dtype spatial_scale,
    const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    const Dtype* bottom_rois,
    const int output_dim,
    const int group_size,
    Dtype* top_data,
    int* mapping_channel) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      // The output is in order (n, ctop, ph, pw)
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int ctop = (index / pooled_width / pooled_height) % output_dim;
      int n = index / pooled_width / pooled_height / output_dim;

      // [start, end) interval for spatial sampling
      bottom_rois += n * 5;
      int roi_batch_ind = bottom_rois[0];
      Dtype roi_start_w =
        static_cast<Dtype>(bottom_rois[1]) * spatial_scale;
      Dtype roi_start_h =
        static_cast<Dtype>(bottom_rois[2]) * spatial_scale;
      Dtype roi_end_w =
        static_cast<Dtype>(bottom_rois[3]) * spatial_scale;
      Dtype roi_end_h =
        static_cast<Dtype>(bottom_rois[4]) * spatial_scale;

      // Force too small ROIs to be 1x1
      Dtype roi_width = max(roi_end_w - roi_start_w, ((Dtype)0.0));  // avoid 0
      Dtype roi_height = max(roi_end_h - roi_start_h, ((Dtype)0.0));

      // Compute w and h at bottom
      Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height);
      Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width);

      Dtype hstart = static_cast<Dtype>(ph) * bin_size_h+ roi_start_h;
      Dtype wstart = static_cast<Dtype>(pw)* bin_size_w+ roi_start_w;
      Dtype hend = static_cast<Dtype>(ph + 1) * bin_size_h+ roi_start_h;
      Dtype wend = static_cast<Dtype>(pw + 1) * bin_size_w+ roi_start_w;
      
      // Add roi offsets and clip to input boundaries
      // hstart = min(max(hstart, 0), height);
      // hend = min(max(hend, 0), height);
      // wstart = min(max(wstart, 0), width);
      // wend = min(max(wend, 0), width);
      // bool is_empty = (hend <= hstart) || (wend <= wstart);

      int gw = pw;
      int gh = ph;
      int c = (ctop*group_size + gh)*group_size + gw;

      bottom_data += (roi_batch_ind * channels + c) * height * width;
      
      Dtype bin_area = max(Dtype(0.0), (hend - hstart)*(wend - wstart));
      mapping_channel[index] = c;
      
      if (bin_area == Dtype(0)) {
        top_data[index] = Dtype(0);
        return;
      }

      Dtype out_sum = 0;      
      int s_w, s_h, e_w, e_h;
    
      s_w = floorf(wstart);
      e_w = ceilf(wend);
      s_h = floorf(hstart);
      e_h = ceilf(hend);
      
      for (int w_iter = s_w; w_iter < e_w; ++w_iter){
        for (int h_iter = s_h; h_iter < e_h; ++h_iter){
          //int bottom_index = h*width + w;
          out_sum += PrRoIPoolingMatCalculation(bottom_data, h_iter, w_iter, h_iter + 1, w_iter + 1, 
                max(hstart, Dtype(h_iter)), max(wstart, Dtype(w_iter)),
                min(hend, Dtype(h_iter) + (Dtype)1.0), min(wend, Dtype(w_iter + (Dtype)1.0)), height, width);
        }
      }

      top_data[index] = out_sum/bin_area;
    }
  }

  template <typename Dtype>
  void PSPRROIPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* bottom_rois = bottom[1]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    int* mapping_channel_ptr = mapping_channel_.mutable_gpu_data();
    int count = top[0]->count();
    caffe_gpu_set(count, Dtype(0), top_data);
    caffe_gpu_set(count, -1, mapping_channel_ptr);
    // NOLINT_NEXT_LINE(whitespace/operators)
    PSPRROIPoolingForward<Dtype> << <CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS >> >(count, bottom_data, spatial_scale_,
      channels_, height_, width_, pooled_height_,
      pooled_width_, bottom_rois, output_dim_, group_size_,
      top_data, mapping_channel_ptr);
    CUDA_POST_KERNEL_CHECK;
  }

  template <typename Dtype>
  __global__ void PSPRROIPoolingBackwardAtomic(
    const int nthreads,
    const Dtype* top_diff,
    const int* mapping_channel,
    const int num_rois,
    const Dtype spatial_scale,
    const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    const int output_dim,
    Dtype* bottom_diff,
    const Dtype* bottom_rois) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      // The output is in order (n, ctop, ph, pw)
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int n = index / pooled_width / pooled_height / output_dim;

      // [start, end) interval for spatial sampling
      bottom_rois += n * 5;
      int roi_batch_ind = bottom_rois[0];
      Dtype roi_start_w =
        static_cast<Dtype>(bottom_rois[1]) * spatial_scale;
      Dtype roi_start_h =
        static_cast<Dtype>(bottom_rois[2]) * spatial_scale;
      Dtype roi_end_w =
        static_cast<Dtype>(bottom_rois[3]) * spatial_scale;
      Dtype roi_end_h =
        static_cast<Dtype>(bottom_rois[4]) * spatial_scale;

      // Force too small ROIs to be 1x1
      Dtype roi_width = max(roi_end_w - roi_start_w, (Dtype)0);  // avoid 0
      Dtype roi_height = max(roi_end_h - roi_start_h, (Dtype)0);

      // Compute w and h at bottom
      Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height);
      Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width);

      Dtype hstart = static_cast<Dtype>(ph)* bin_size_h
        + roi_start_h;
      Dtype wstart = static_cast<Dtype>(pw)* bin_size_w
        + roi_start_w;
      Dtype hend = static_cast<Dtype>(ph + 1) * bin_size_h
        + roi_start_h;
      Dtype wend = static_cast<Dtype>(pw + 1) * bin_size_w
        + roi_start_w;
      // Add roi offsets and clip to input boundaries
      // hstart = min(max(hstart, 0), height);
      // hend = min(max(hend, 0), height);
      // wstart = min(max(wstart, 0), width);
      // wend = min(max(wend, 0), width);
      // bool is_empty = (hend <= hstart) || (wend <= wstart);

      // Compute c at bottom
      int c = mapping_channel[index];
      Dtype* offset_bottom_diff = bottom_diff +
        (roi_batch_ind * channels + c) * height * width;
        
        
      Dtype bin_area = max(Dtype(0.0), (hend - hstart)*(wend - wstart));
      Dtype diff_val = top_diff[index];
      Dtype sum_out = bin_area == Dtype(0) ? Dtype(0) : diff_val / bin_area;
      
      int s_w, s_h, e_w, e_h;

      s_w = floorf(wstart);
      e_w = ceilf(wend);
      s_h = floorf(hstart);
      e_h = ceilf(hend);
      for (int w_iter = s_w; w_iter < e_w; ++w_iter)
        for (int h_iter = s_h; h_iter < e_h; ++h_iter)
            PrRoIPoolingMatDistributeDiff(offset_bottom_diff, sum_out, h_iter, w_iter, h_iter + 1, w_iter + 1, 
                max(hstart, Dtype(h_iter)), max(wstart, Dtype(w_iter)),
                min(hend, Dtype(h_iter) + (Dtype)1.0), min(wend, Dtype(w_iter + (Dtype)1.0)),
                height, width);
    }
  }

  template <typename Dtype>
  void PSPRROIPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (!propagate_down[0]) {
      return;
    }

    const Dtype* bottom_rois = bottom[1]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int bottom_count = bottom[0]->count();
    const int* mapping_channel_ptr = mapping_channel_.gpu_data();
    caffe_gpu_set(bottom[1]->count(), Dtype(0), bottom[1]->mutable_gpu_diff());
    caffe_gpu_set(bottom_count, Dtype(0), bottom_diff);
    const int count = top[0]->count();
    // NOLINT_NEXT_LINE(whitespace/operators)
    PSPRROIPoolingBackwardAtomic<Dtype> << <CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS >> >(count, top_diff, mapping_channel_ptr,
      top[0]->num(), spatial_scale_, channels_, height_, width_,
      pooled_height_, pooled_width_, output_dim_, bottom_diff,
      bottom_rois);
    CUDA_POST_KERNEL_CHECK;
  }

  INSTANTIATE_LAYER_GPU_FUNCS(PSPRROIPoolingLayer);

}  // namespace caffe
