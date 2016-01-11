#include <algorithm>
#include <vector>

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/crop_layer.hpp"
namespace caffe {

template <typename Dtype>
void CropLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
      int channel_axis = 1;
      first_spatial_axis_ = channel_axis + 1;
      const int num_axes = bottom[0]->num_axes();
      num_spatial_axes_ = num_axes - first_spatial_axis_;
      vector<int> spatial_dim_blob_shape(1,std::max(num_spatial_axes_, 1));
      crop_shape_.Reshape(spatial_dim_blob_shape);
      src_shape_.Reshape(spatial_dim_blob_shape);
  // Construct a map from top blobs to layer inds, skipping over in-place
  // connections.

  // map<Blob<Dtype>*, int> down_map;
  // for (int layer_ind = 0; layer_ind < this->net_->top_vecs().size();
  //      ++layer_ind) {
  //   vector<Blob<Dtype>*> tops = this->net_->top_vecs()[layer_ind];
  //   for (int top_ind = 0; top_ind < tops.size(); ++top_ind) {
  //     if (down_map.find(tops[top_ind]) == down_map.end()) {
  //       down_map[tops[top_ind]] = layer_ind;
  //     }
  //   }
  // }
  //
  //
  //
  // // Walk back from the first bottom, keeping track of all the blobs we pass.
  // set<Blob<Dtype>*> path_blobs;
  // Blob<Dtype>* blob = bottom[0];
  // int layer_ind;
  // // TODO this logic can be simplified if all blobs are tops
  // path_blobs.insert(blob);
  // while (down_map.find(blob) != down_map.end()) {
  //   layer_ind = down_map[blob];
  //   if (this->net_->bottom_vecs()[layer_ind].size() == 0) {
  //     break;
  //   }
  //   blob = this->net_->bottom_vecs()[layer_ind][0];
  //   path_blobs.insert(blob);
  // }
  // // Now walk back from the second bottom, until we find a blob of intersection.
  // Blob<Dtype>* inter_blob = bottom[1];
  // while (path_blobs.find(inter_blob) == path_blobs.end()) {
  //   CHECK(down_map.find(inter_blob) != down_map.end())
  //       << "Cannot align apparently disconnected blobs.";
  //   layer_ind = down_map[inter_blob];
  //   CHECK_GT(this->net_->bottom_vecs()[layer_ind].size(), 0)
  //       << "Cannot align apparently disconnected blobs.";
  //   inter_blob = this->net_->bottom_vecs()[layer_ind][0];
  // }
  // Compute the coord map from the blob of intersection to each bottom.
//  vector<DiagonalAffineMap<Dtype> > coord_maps(2,
  //    DiagonalAffineMap<Dtype>::identity(2));
  // for (int i = 0; i < 2; ++i) {
  //   for (Blob<Dtype>* blob = bottom[i]; blob != inter_blob;
  //        blob = this->net_->bottom_vecs()[down_map[blob]][0]) {
  //     shared_ptr<Layer<Dtype> > layer = this->net_->layers()[down_map[blob]];
  //   //  coord_maps[i] = coord_maps[i].compose(layer->coord_map());
  //   }
  // }

}

template <typename Dtype>
void CropLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
      int* crop_shape_data = crop_shape_.mutable_cpu_data();
      int* src_shape_data = src_shape_.mutable_cpu_data();
      vector<int> shape_0=bottom[0]->shape();
      vector<int> shape_1=bottom[1]->shape();
      for(int i=0;i<num_spatial_axes_;++i){
        crop_shape_data[i]= shape_1[first_spatial_axis_+i];
        src_shape_data[i] = shape_0[first_spatial_axis_+i];
      }
      shape_1[0]=bottom[0]->num();
      shape_1[1]=bottom[0]->channels();
      top[0]->Reshape(shape_1);
}

template <typename Dtype>
void CropLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int n = 0; n < top[0]->num(); ++n) {
    for (int c = 0; c < top[0]->channels(); ++c) {
      for (int h = 0; h < top[0]->height(); ++h) {
        caffe_copy(top[0]->width(),
            bottom_data + bottom[0]->offset(n, c, crop_h_ + h, crop_w_),
            top_data + top[0]->offset(n, c, h));
      }
    }
  }
}

template <typename Dtype>
void CropLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  if (propagate_down[0]) {
    caffe_set(bottom[0]->count(), static_cast<Dtype>(0), bottom_diff);
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < top[0]->channels(); ++c) {
        for (int h = 0; h < top[0]->height(); ++h) {
          caffe_copy(top[0]->width(),
              top_diff + top[0]->offset(n, c, h),
              bottom_diff + bottom[0]->offset(n, c, crop_h_ + h, crop_w_));
        }
      }
    }
  }
}


  #ifdef CPU_ONLY
  STUB_GPU(CropLayer);
  #endif


  INSTANTIATE_CLASS(CropLayer);
  REGISTER_LAYER_CLASS(Crop);
}
