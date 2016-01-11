#include <vector>

#include "caffe/layers/crop_layer.hpp"

namespace caffe {

// Copy (one line per thread) from one array to another, with arbitrary
// strides in the last two dimensions.
template <typename Dtype>
__global__ void copy_kernel(const int n, const int height, const int width,
    const int src_outer_stride, const int src_inner_stride,
    const int dest_outer_stride, const int dest_inner_stride,
    const Dtype* src, Dtype* dest) {
  CUDA_KERNEL_LOOP(index, n) {
    int src_start = index / height * src_outer_stride
                  + index % height * src_inner_stride;
    int dest_start = index / height * dest_outer_stride
                   + index % height * dest_inner_stride;
    for (int i = 0; i < width; ++i) {
      dest[dest_start + i] = src[src_start + i];
    }
  }
}


// Copy (one line per thread) from one array to another, with arbitrary
// strides in the last two dimensions.
template <typename Dtype, int num_axes>
__global__ void crop_kernel_ND(const int n, const int channels,const int* dest_shape,
    const int* source_shape,  const Dtype* src, Dtype* dest) {
      int d_dest_pt[num_axes];  // NOLINT(runtime/arrays)
      //int d_iter[num_axes];  // NOLINT(runtime/arrays)
    //  int d_src_pt[num_axes];
      int i,j;
  CUDA_KERNEL_LOOP(index, n) {
    int channel_in = index;
  //  int channel_out = 1;
    int n  =0;
    int c =0;
    //int kernel_length =1;
    for (i = num_axes - 1; i >= 0; --i) {
      d_dest_pt[i] = channel_in % dest_shape[i];
      channel_in /= dest_shape[i];
    //  channel_out *= kernel[i];
     }
      c = channel_in%channels;
      n =channel_in/channels;
    int nc_len =(n * channels + c);
    int src_dim_len=1;
    for(i=0;i<num_axes;++i){
      src_dim_len*=source_shape[i];
    }
    const Dtype* const src_slice = src + nc_len * src_dim_len;

    int src_data_idx =d_dest_pt[0];
    bool out_range_source =false;
    //Dtype  point_value ;
    for(j=1; j<num_axes; ++j){
        src_data_idx*=source_shape[j];
        src_data_idx+=d_dest_pt[j];
        // point_value =
        //   d_dest_pt[j]> source_shape[j]-1?
        //   Dtype(0):
        //   src_slice[src_data_idx];
        if(d_dest_pt[j]> source_shape[j]-1)
          {
            out_range_source  =true;
            break;
          }
      }

      dest[n]= out_range_source? Dtype(0):src_slice[src_data_idx];
  }
}


template <typename Dtype>
void CropLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int lines = top[0]->count();
  const int*  dest_shape =crop_shape_.gpu_data();
  const int*  source_shape =src_shape_.gpu_data();

  switch(num_spatial_axes_){
    case 1:
    crop_kernel_ND<Dtype,1 ><<<CAFFE_GET_BLOCKS(lines), CAFFE_CUDA_NUM_THREADS>>>(
      lines, bottom[0]->channels(),dest_shape,
      source_shape,  bottom_data, top_data);
      break;
    case 2:
    crop_kernel_ND<Dtype,2 ><<<CAFFE_GET_BLOCKS(lines), CAFFE_CUDA_NUM_THREADS>>>(
      lines, bottom[0]->channels(),dest_shape,
      source_shape,  bottom_data, top_data);
      break;
    case 3:
      crop_kernel_ND<Dtype,3 ><<<CAFFE_GET_BLOCKS(lines), CAFFE_CUDA_NUM_THREADS>>>(
        lines, bottom[0]->channels(),dest_shape,
        source_shape,  bottom_data, top_data);
        break;
    case 4:
      crop_kernel_ND<Dtype,3 ><<<CAFFE_GET_BLOCKS(lines), CAFFE_CUDA_NUM_THREADS>>>(
          lines, bottom[0]->channels(),dest_shape,
          source_shape,  bottom_data, top_data);
          break;
    default:
          LOG(FATAL) << "Unsupported crop dimension.";
    }
  // NOLINT_NEXT_LINE(whitespace/operators)
  // copy_kernel<<<CAFFE_GET_BLOCKS(lines), CAFFE_CUDA_NUM_THREADS>>>(
  //     lines, top[0]->height(), top[0]->width(),
  //     bottom[0]->height() * bottom[0]->width(), bottom[0]->width(),
  //     top[0]->height() * top[0]->width(), top[0]->width(),
  //     bottom_data + bottom[0]->offset(0, 0, crop_h_, crop_w_), top_data);
}

template <typename Dtype>
void CropLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int lines = top[0]->count() ;/// top[0]->width();
  const int*  source_shape =crop_shape_.gpu_data();
  const int*  dest_shape =src_shape_.gpu_data();

  if (propagate_down[0]) {
    caffe_gpu_set(bottom[0]->count(), static_cast<Dtype>(0), bottom_diff);
    // NOLINT_NEXT_LINE(whitespace/operators)
    switch(num_spatial_axes_){
      case 1:
      crop_kernel_ND<Dtype,1 ><<<CAFFE_GET_BLOCKS(lines), CAFFE_CUDA_NUM_THREADS>>>(
        lines, bottom[0]->channels(),dest_shape,
        source_shape,  top_diff, bottom_diff);
        break;
      case 2:
        crop_kernel_ND<Dtype,2 ><<<CAFFE_GET_BLOCKS(lines), CAFFE_CUDA_NUM_THREADS>>>(
          lines, bottom[0]->channels(),dest_shape,
          source_shape,  top_diff, bottom_diff);
          break;
      case 3:
        crop_kernel_ND<Dtype,3 ><<<CAFFE_GET_BLOCKS(lines), CAFFE_CUDA_NUM_THREADS>>>(
            lines, bottom[0]->channels(),dest_shape,
            source_shape,  top_diff, bottom_diff);
            break;
      case 4:
        crop_kernel_ND<Dtype,4 ><<<CAFFE_GET_BLOCKS(lines), CAFFE_CUDA_NUM_THREADS>>>(
            lines, bottom[0]->channels(),dest_shape,
            source_shape,  top_diff, bottom_diff);
            break;
      default:
              LOG(FATAL) << "Unsupported crop dimension.";
          }

   }
}

INSTANTIATE_LAYER_GPU_FUNCS(CropLayer);

}  // namespace caffe
