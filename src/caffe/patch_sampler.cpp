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
map<const string, weak_ptr<Runner<float> > > runners_;
static boost::mutex runners_mutex_;

template <typename Dtype>
PatchSampler<Dtype>::PatchSampler(const LayerParameter& param)
    :queue_pair_(new QueuePair_Batch<Dtype>(param)) {
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

//
template <typename Dtype>
QueuePair_Batch<Dtype>::QueuePair_Batch(const LayerParameter& param) {
  // Initialize the free queue with requested number of blobs
  //shape1[0]=size;
  // int num  =1;
  // Blob d1= new Blob<Dtype>(shape1);
  // Blob d2= new Blob<Dtype>(shape2);
  // Blob d2= new Blob<Dtype>(shape3);
  // Blob l = new Blob<Dtype>(shape_l);
  // Batch_data<> bm;
  // bm.data_.push_back(d1);
  // bm.data_.push_back(d2);
  // bm.data_.push_back(d3);
  // bm.label =l;
  // for (int i = 0; i < size; ++i) {
  //   free_.push(bm);
  // }
}

template <typename Dtype>
QueuePair_Batch<Dtype>::~QueuePair_Batch() {
  //Datum* datum;
  // while (free_.try_pop(&datum)) {
  //   delete datum;
  // }
  // while (full_.try_pop(&datum)) {
  //   delete datum;
  // }
}

//
template <typename Dtype>
Runner<Dtype>::Runner(const LayerParameter& param)
    : param_(param),
      new_queue_pairs_() {
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
