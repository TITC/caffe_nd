#include <string>
#include <vector>

#include "caffe/data_transformerND.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<typename Dtype>
DataTransformerND<Dtype>::DataTransformerND(const TransformationNDParameter& param)
    : param_(param){
  // check if we want to use mean_value
  if (param_.mean_value_size() > 0) {
    for (int c = 0; c < param_.mean_value_size(); ++c) {
      mean_values_.push_back(param_.mean_value(c));
    }
  }
}

template<typename Dtype>
const CropCenterInfo<Dtype>  DataTransformerND<Dtype>::PeekCropCenterPoint(Blob<Dtype>* input_blob){
  //PeekCropCenterPoint assum that input is a label array, thus it has only one channel.

  //for(int i=0;input_blob->shape().size();++i)
    //  LOG(INFO)<<"input blob shape size =" <<input_blob->shape().size();
  bool padding =param_.padding();
  bool crop =param_.has_crop_shape();
  int input_shape_dims=input_blob->num_axes();
  CropCenterInfo<Dtype> crop_center_info;//(new CropCenterInfo<Dtype>);
  //CropCenterInfo<Dtype> crop_center_info;//(new CropCenterInfo<Dtype>);
  CHECK_GE(input_shape_dims,3);
  const vector<int>& input_shape =input_blob->shape();
  vector<int> tranform_shape;
  const int input_num = input_shape[0];
  const int input_channels = input_shape[1];
  CHECK_EQ(input_channels,1);
  CHECK_EQ(input_num,1);
  int crop_shape_dims=param_.crop_shape().dim_size();
  CHECK_EQ(crop_shape_dims,input_shape_dims-2);
  // assume that the number
  tranform_shape.push_back(1);
  tranform_shape.push_back(1);
  CHECK_EQ(crop_shape_dims,input_shape_dims-2);
  for(int i=0;i<crop_shape_dims;i++){
      tranform_shape.push_back(param_.crop_shape().dim(i));
  }
  vector<int> nd_off(crop_shape_dims,0);

     for(int i=0;i<nd_off.size();i++){
       if(crop)
          if(padding){
              CHECK_GE(input_shape[i+2],0);
              nd_off[i] = Rand(input_shape[i+2])-tranform_shape[i]/2;
            }
          else
          {
              nd_off[i] = Rand(input_shape[i+2] - tranform_shape[i] + 1);
              CHECK_GE(input_shape[i+2],0);
            }
        else
          nd_off[i] = (input_shape[i+2]  - tranform_shape[i]) / 2;

          //for(int i=0;input_blob->shape().size();++i)
        //LOG(INFO)<<"patch offset  =" <<nd_off[i] ;
      //  w_off = (input_width - crop_size) / 2;
    }



  int center_index=1;
  for (int n=0;n<nd_off.size();++n){
       if(n==0){
         center_index = nd_off[n];
          // LOG(INFO)<<"n=0 center inx "<<center_index;
       }else{
       center_index*=input_shape[2+n];
       center_index+=nd_off[n];
     }
  }
  int input_count=input_blob->count();
  CHECK_GE(center_index,0);
  CHECK_LE(center_index,input_count);
  const Dtype* input_data =input_blob->cpu_data();
  Dtype center_value=input_data[center_index];
  crop_center_info.nd_off=nd_off;
  crop_center_info.value =center_value;

  //LOG(INFO)<<"nd_off num_aix ="<<crop_center_info.nd_off.size();
  return crop_center_info;
  //TransformationNDParameter_PadMethod_ZERO;

}

template<typename Dtype>
void DataTransformerND<Dtype>::Transform(Blob<Dtype>* input_blob,
                                                Blob<Dtype>* transformed_blob){
//  CropCenterInfo<Dtype> c_info= PeekCropCenterPoint(input_blob);

  CropCenterInfo<Dtype> c_info= PeekCropCenterPoint(input_blob);
  Transform(input_blob, transformed_blob,c_info.nd_off);

}
template<typename Dtype>
void DataTransformerND<Dtype>::Transform(Blob<Dtype>* input_blob,
                                       Blob<Dtype>* transformed_blob, const vector<int>& off_set) {
  int input_shape_dims=input_blob->num_axes();
  bool padding =param_.padding();
  int offset_axis = off_set.size();
  CHECK_GE(input_shape_dims,3);
  CHECK_EQ(offset_axis,input_shape_dims-2);
  bool crop =param_.has_crop_shape();
  const vector<int>& input_shape =input_blob->shape();
  vector<int> transform_shape;
  const int input_num = input_shape[0];
  const int input_channels = input_shape[1];
  transform_shape.push_back(input_num);
  transform_shape.push_back(input_channels);
  int crop_shape_dims=param_.crop_shape().dim_size();
  CHECK_EQ(crop_shape_dims,input_shape_dims-2);
  for(int i=0;i<crop_shape_dims;i++){
      transform_shape.push_back(param_.crop_shape().dim(i));
  }

  if (transformed_blob->count() == 0) {
    // Initialize transformed_blob with the right shape.
    if (crop) {
      transformed_blob->Reshape(transform_shape);
    } else {
      transformed_blob->Reshape(input_shape);
    }
  }
 //return ;
  vector<int> new_transform_shape = transformed_blob->shape();
  //const int num = new_transform_shape[0];
  const int channels = new_transform_shape[1];
  // const int height = transformed_blob->height();
  // const int width = transformed_blob->width();
  const size_t trans_data_size = transformed_blob->count();

  //CHECK_LE(input_num, num);
  CHECK_EQ(input_channels, channels);
  //CHECK_GE(input_height, height);
  //CHECK_GE(input_width, width);


  const Dtype scale = param_.scale();
  // do mirro for each of dimention respectively
  const bool do_mirror = param_.mirror() && Rand(crop_shape_dims+1);
  const bool has_mean_values = mean_values_.size() > 0;

  //int h_off = 0;
  //int w_off = 0;
  vector<int> nd_off(crop_shape_dims,0);

  if(crop){
    nd_off=off_set;
  }
  // if (crop) {
  //   //CHECK_EQ(crop_size, height);
  //   //CHECK_EQ(crop_size, width);
  //   // We only do random crop when we do training.
  //    for(int i=0;i<nd_off.size();i++){
  //     nd_off[i] = Rand(input_shape[i+2] - tranform_shape[i] + 1);
  //   }
  //   //  w_off = Rand(input_width - crop_size + 1);
  // } //else {
    //CHECK_EQ(input_height, height);
    //CHECK_EQ(input_width, width);
  //}

  // Dtype* input_data = input_blob->mutable_cpu_data();
  // if (has_mean_values) {
  //   CHECK(mean_values_.size() == 1 || mean_values_.size() == input_channels) <<
  //    "Specify either 1 mean_value or as many as channels: " << input_channels;
  //   if (mean_values_.size() == 1) {
  //     caffe_add_scalar(input_blob->count(), -(mean_values_[0]), input_data);
  //   } else {
  //     for (int n = 0; n < input_num; ++n) {
  //       for (int c = 0; c < input_channels; ++c) {
  //         int offset = input_blob->offset(n, c);
  //         caffe_add_scalar(input_height * input_width, -(mean_values_[c]),
  //           input_data + offset);
  //       }
  //     }
  //   }
  // }

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  //tranform_shape
  //int top_index  =0;
//  int data_index =0;
  //int input_num  = input_shape(0);
  //int channels   = input_shape(1);
  int start_spatial_aixs =2;

 for(size_t p=0;p<trans_data_size;++p){
   // revise compute the dat index in the input blob;
   vector<int> nd_point;
   vector<int>::iterator it;
   size_t pre_aixs_len =0;
   for(int i=transform_shape.size()-1;i>-1;--i){
      int data_axis_idx =0;
     if(i==transform_shape.size()-1){
        data_axis_idx=p%transform_shape[i]+off_set[i-2];
        //if(do_mirror)
        //    transform_shape[i]-(data_axis_idx+1);
        //nd_point.push_back(data_axis_idx);
        pre_aixs_len=transform_shape[i];
     }else{
        //if(i-2>=0)
          data_axis_idx= i-2>=0 ? p/pre_aixs_len +off_set[i-2]:p/pre_aixs_len;
        //  nd_point.push_back(data_axis_idx);
          pre_aixs_len*=transform_shape[i];
     }
       it =nd_point.begin();
       nd_point.insert(it, data_axis_idx);
       //nd_point.push_back(data_axis_idx);
   }

  // transformed_data[p]=input_blob.data_at();
   //read the data from the input blob to transformed bloab;
  size_t data_idx=0;

   for (int n=0;n<nd_point.size();++n){
        if(n==0){
          data_idx = nd_point[n];
        }else{
        data_idx*=input_shape[n];
        data_idx+=nd_point[n];
      }
   }

   size_t input_count=input_blob->count();
   const Dtype* input_data =input_blob->cpu_data();
   bool data_in_pad_space =(data_idx>=0 && data_idx<input_count);
   if(data_in_pad_space)
     transformed_data[p]=input_data[data_idx];
   else
     transformed_data[p]=0;
 }




//
//   vctor<int > cur_point;
//
//   vctor<vector<int>> all_points_in_intrans(trans_data_size);
// //  for(int i=0;i<trans_data_size;++i){
// //       vector<int> point(tranform_shape.size()) ;
//       for (int n=0;n<tranform_shape.size();++n){
//           int size =tranform_shape[n];
//
//
//       }
//         (all_points_in_intrans[i])[n];
// //  }
//   for (int n=0;n<tranform_shape.size();++n){
//      //if(n==0) data_index = tranform_shape[];
//      //for(int i=)
//      //int count_trans =
//   }
//
//
//   int center_index=1;
//   for (int n=0;n<nd_off.size();++n){
//        if(n==0){
//          center_index = input_shape[2+n] *tranform_shape[n];
//        }else{
//        center_index*=input_shape[2+n];
//        center_index+=(nd_off[n]+tranform_shape[n]/2);
//      }
//   }
//
//   for (int n = 0; n < input_num; ++n) {
//     int top_index_n = n * channels;
//     int data_index_n = n * channels;
//     for (int c = 0; c < channels; ++c) {
//       int top_index_c = (top_index_n + c) * height;
//       int data_index_c = (data_index_n + c) * input_height + h_off;
//
//   for (int n = 0; n < input_num; ++n) {
//     int top_index_n = n * channels;
//     int data_index_n = n * channels;
//     for (int c = 0; c < channels; ++c) {
//       int top_index_c = (top_index_n + c) * height;
//       int data_index_c = (data_index_n + c) * input_height + h_off;
//       for (int h = 0; h < height; ++h) {
//         int top_index_h = (top_index_c + h) * width;
//         int data_index_h = (data_index_c + h) * input_width + w_off;
//         if (do_mirror) {
//           int top_index_w = top_index_h + width - 1;
//           for (int w = 0; w < width; ++w) {
//             transformed_data[top_index_w-w] = input_data[data_index_h + w];
//           }
//         } else {
//           for (int w = 0; w < width; ++w) {
//             transformed_data[top_index_h + w] = input_data[data_index_h + w];
//           }
//         }
//       }
//     }
//   }



  if (scale != Dtype(1)) {
    DLOG(INFO) << "Scale: " << scale;
    caffe_scal( trans_data_size, scale, transformed_data);
  }
}

// template<typename Dtype>
// vector<int> DataTransformer<Dtype>::InferBlobShape(const Datum& datum) {
//   if (datum.encoded()) {
// #ifdef USE_OPENCV
//     CHECK(!(param_.force_color() && param_.force_gray()))
//         << "cannot set both force_color and force_gray";
//     cv::Mat cv_img;
//     if (param_.force_color() || param_.force_gray()) {
//     // If force_color then decode in color otherwise decode in gray.
//       cv_img = DecodeDatumToCVMat(datum, param_.force_color());
//     } else {
//       cv_img = DecodeDatumToCVMatNative(datum);
//     }
//     // InferBlobShape using the cv::image.
//     return InferBlobShape(cv_img);
// #else
//     LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
// #endif  // USE_OPENCV
//   }
//   const int crop_size = param_.crop_size();
//   const int datum_channels = datum.channels();
//   const int datum_height = datum.height();
//   const int datum_width = datum.width();
//   // Check dimensions.
//   CHECK_GT(datum_channels, 0);
//   CHECK_GE(datum_height, crop_size);
//   CHECK_GE(datum_width, crop_size);
//   // Build BlobShape.
//   vector<int> shape(4);
//   shape[0] = 1;
//   shape[1] = datum_channels;
//   shape[2] = (crop_size)? crop_size: datum_height;
//   shape[3] = (crop_size)? crop_size: datum_width;
//   return shape;
// }

template <typename Dtype>
void DataTransformerND<Dtype>::InitRand() {
  const bool needs_rand = param_.has_crop_shape();
  if (needs_rand) {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  } else {
    rng_.reset();
  }
}

template <typename Dtype>
int DataTransformerND<Dtype>::Rand(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}

INSTANTIATE_CLASS(DataTransformerND);

}  // namespace caffe
