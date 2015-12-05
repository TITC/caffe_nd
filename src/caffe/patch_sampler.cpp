#include <boost/thread.hpp>
#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/patch_sampler.hpp"
//#include "caffe/layers/data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

using boost::weak_ptr;

//template <typename Dtype>
//map<const string, weak_ptr<Runner<float> > > runners_;
//map<const string, weak_ptr<Runner<double> > > runners_;
static boost::mutex runners_mutex_;

template <typename Dtype>
PatchSampler<Dtype>::PatchSampler(const LayerParameter& param)
    :param_(param),
    queue_pair_(new QueuePair_Batch<Dtype>(param)),
    d_provider_(new Data_provider<Dtype>(param)),
    data_transformer_nd(new DataTransformerND<Dtype>(param.transform_nd_param()))
{
  // Get or create a body
  boost::mutex::scoped_lock lock(runners_mutex_);
  string key = source_key(param);
  weak_ptr<Runner<Dtype> >& weak = runners_[key];
  runner_ = weak.lock();
  if (!runner_) {
    runner_.reset(new Runner<Dtype>(param));
    runners_[key] = weak_ptr<Runner<Dtype> >(runner_);
  }
  runner_->new_queue_pairs_.push(queue_pair_);

  patches_per_data_batch_ = param_.patch_sampler_pram().patches_per_data_batch();
  const unsigned int prefetch_rng_seed = caffe_rng_rand();
  prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
  patch_count_ =0;
}

template <typename Dtype>
PatchSampler<Dtype>::~PatchSampler() {
  string key = source_key(runner_->param_);
  runner_.reset();
  boost::mutex::scoped_lock lock(runners_mutex_);
  if (runners_[key].expired()) {
    runners_.erase(key);
  }
}

template <typename Dtype>
unsigned int PatchSampler<Dtype>::PrefetchRand(){
  CHECK(prefetch_rng_);
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  return (*prefetch_rng)();
}

template <typename Dtype>
void PatchSampler<Dtype>::ReadOnePatch( QueuePair_Batch<Dtype>* qb ){
  // load new data to memomry pool
  if(patch_count_%patches_per_data_batch_ ==0)
  {
    d_provider_->load_next_batch();
  }
  patch_count_++;
  int data_idx=PrefetchRand()% d_provider_->get_current_batch_size();
  const Batch_data<Dtype> source_data_label=d_provider_->getOneData(data_idx);
  Batch_data<Dtype>* patch_data_label = qb->free_.pop();
  //TODO
  // take input patch_data and then warp a patch and put it to patch_data;
  // the cappablity that address the probability of selecting classe need to be addressed
   //CropCenterInfo<Dtype>& PeekCropCenterPoint(Blob<Dtype>* input_blob);
  CropCenterInfo<Dtype>& crp_cent_info=data_transformer_nd->PeekCropCenterPoint(&source_data_label.label);
  Blob<Dtype> trans_blob;
  data_transformer_nd->Transform(&source_data_label.data, &trans_blob, crp_cent_info.nd_off);
  vector<int>& source_data_shape =source_data_label.data.shape();
  vector<int>& source_label_shape =source_data_label.label.shape();
  int d_dims  =source_data_shape.size();
  int l_dims  =source_label_shape.size();
  int d_num   =source_data_shape[0];
  int l_num   =source_label_shape[0];
  CHECK_EQ(d_dims,l_dims);
  CHECK_EQ(d_num,1);
  CHECK_EQ(l_num,1);
  for(int i=2;i<d_dims;i++){
    int d_dim=source_data_shape[i];
    int l_dim=source_label_shape[i];
    CHECK_EQ(d_dim,l_dim);
  }






  //vector<int>& source_shape =patch_data->data.shape();
//  qb->full_.push(patch_data);

  // go to the next iter
  // cursor->Next();
  // if (!cursor->valid()) {
  //   DLOG(INFO) << "Restarting data prefetching from start.";
  //   cursor->SeekToFirst();
  // }

}
//
template <typename Dtype>
QueuePair_Batch<Dtype>::QueuePair_Batch(const LayerParameter& param) {
  // Initialize the free queue with requested number of blobs
//  Batch_dat
  int batch_size = param.patch_sampler_pram().batch_size();
  int patch_dims = param.patch_sampler_pram().patch_shape().dim_size();
  vector<int> patch_shape, label_shape;
  patch_shape.push_back(batch_size);
  label_shape.push_back(batch_size);
  for(int i=0;i<patch_dims;i++){
    int dim =param.patch_sampler_pram().patch_shape().dim(i);
    patch_shape.push_back(dim);
  }

  bool output_label =param.patch_sampler_pram().has_label_shape();
  if(output_label){
    int label_dim = param.patch_sampler_pram().label_shape().dim_size();
    for(int i=1;i<label_dim;i++){
       label_shape.push_back(label_dim);
     }
  }

  for (int i = 0; i < batch_size; ++i) {
    free_.push(new Batch_data<Dtype>);
    Batch_data<Dtype>& b_data = *( free_->peek());
    b_data.data.Reshape(patch_shape);
    if(output_label){
      b_data.label.Reshape(patch_shape);
    }
  }
}

template <typename Dtype>
QueuePair_Batch<Dtype>::~QueuePair_Batch() {
  Batch_data<Dtype>* bd;
  while (free_.try_pop(&bd)) {
    delete bd;
  }
  while (full_.try_pop(&bd)) {
    delete bd;
  }
}

//
template <typename Dtype>
Runner<Dtype>::Runner(const LayerParameter& param,const PatchSampler<Dtype>& p_sampler)
    : param_(param),
      new_queue_pairs_(),
      p_sampler_(p_sampler)
      {
  StartInternalThread();
}

template <typename Dtype>
Runner<Dtype>::~Runner() {
  StopInternalThread();
}

template <typename Dtype>
void Runner<Dtype>::InternalThreadEntry() {
  // shared_ptr<db::DB> db(db::GetDB(param_.data_param().backend()));
  // db->Open(param_.data_param().source(), db::READ);
  // shared_ptr<db::Cursor> cursor(db->NewCursor());
  // vector<shared_ptr<QueuePair> > qps;
  // try {
  //   int solver_count = param_.phase() == TRAIN ? Caffe::solver_count() : 1;
  //
  //   // To ensure deterministic runs, only start running once all solvers
  //   // are ready. But solvers need to peek on one item during initialization,
  //   // so read one item, then wait for the next solver.
  //   for (int i = 0; i < solver_count; ++i) {
  //     shared_ptr<QueuePair> qp(new_queue_pairs_.pop());
  //     read_one(cursor.get(), qp.get());
  //     qps.push_back(qp);
  //   }
  //   // Main loop
  //   while (!must_stop()) {
  //     for (int i = 0; i < solver_count; ++i) {
  //       read_one(cursor.get(), qps[i].get());
  //     }
  //     // Check no additional readers have been created. This can happen if
  //     // more than one net is trained at a time per process, whether single
  //     // or multi solver. It might also happen if two data layers have same
  //     // name and same source.
  //     CHECK_EQ(new_queue_pairs_.size(), 0);
  //   }
  // } catch (boost::thread_interrupted&) {
  //   // Interrupted exception is expected on shutdown
  // }
}

// void DataReader::Body::read_one(db::Cursor* cursor, QueuePair* qp) {
//   Datum* datum = qp->free_.pop();
//   // TODO deserialize in-place instead of copy?
//   datum->ParseFromString(cursor->value());
//   qp->full_.push(datum);
//
//   // go to the next iter
//   cursor->Next();
//   if (!cursor->valid()) {
//     DLOG(INFO) << "Restarting data prefetching from start.";
//     cursor->SeekToFirst();
//   }
// }

}  // namespace caffe
