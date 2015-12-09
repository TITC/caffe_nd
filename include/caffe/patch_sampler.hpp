#ifndef CAFFE_PATCH_SAMPLER_HPP_
#define CAFFE_DATA_SAMPLER_HPP_

#include <map>
#include <string>
#include <vector>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/util/db.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/data_transformerND.hpp"
#include "caffe/data_provider.hpp"
namespace caffe {
  template <typename Dtype> class PatchSampler;

 template <typename Dtype>
  class QueuePair_Batch {
   public:
    explicit QueuePair_Batch(const LayerParameter& param);
    ~QueuePair_Batch();

    BlockingQueue<Batch_data<Dtype>*> free_;
    BlockingQueue<Batch_data<Dtype>*> full_;

  DISABLE_COPY_AND_ASSIGN(QueuePair_Batch);
  };

  // template <typename Dtype>
  // class Data_provider{
  // public:
  //   explicit Data_provider(const LayerParameter& param);
  //   ~Data_provider();
  //   void load_next_batch(int numData);
  //   inline int get_current_batch_size(){return num_cur_batch_size;};
  //   inline const Batch_data<Dtype>& getOneData(int idx){ return source_data_label_pair_[idx];};
  // protected:
  //   int num_cur_batch_size;
  //   vector<Batch_data<Dtype> > source_data_label_pair_;
  // };

  // A single body is created per source
  template <typename Dtype>
  class Runner : public InternalThread {
   public:
    explicit Runner(const LayerParameter& param, const PatchSampler<Dtype>& p_sampler);
    virtual ~Runner();

   protected:
    void InternalThreadEntry();
    //void read_one(db::Cursor* cursor, QueuePair* qp);

    const LayerParameter param_;
    const PatchSampler<Dtype> p_sampler_;
    //const Data_provider d_provider_;
    BlockingQueue<shared_ptr<QueuePair_Batch<Dtype> > > new_queue_pairs_;

    friend class PatchSampler<Dtype>;

  DISABLE_COPY_AND_ASSIGN(Runner);
  };
/**
 * @brief warp patches from a data_reader_general to queues available to PatchSamplerLayer layers.
 * A single reading thread is created per source, even if multiple solvers
 * are running in parallel, e.g. for multi-GPU training. This makes sure
 * databases are read sequentially, and that each solver accesses a different
 * subset of the database. Data is distributed to solvers in a round-robin
 * way to keep parallel training deterministic.
 */

template <typename Dtype>
class PatchSampler {
 public:
  explicit PatchSampler(const LayerParameter& param);
  ~PatchSampler();

  inline BlockingQueue<Batch_data<Dtype>*>& free() const {
    return queue_pair_->free_;
  }
  inline BlockingQueue<Batch_data<Dtype>*>& full() const {
    return queue_pair_->full_;
  }
  inline vector<int>& patch_data_shape(){
    if (dest_data_shape_.size()==0){
      ReadOnePatch(queue_pair_.get());
    }
    return dest_data_shape_;
  }
  inline vector<int>& patch_label_shape(){
    if (dest_label_shape_.size()==0){
      ReadOnePatch(queue_pair_.get());
    }
    return dest_label_shape_;
  }
 protected:
   void ReadOnePatch(QueuePair_Batch<Dtype>* qb );
   unsigned int PrefetchRand();
  // Queue pairs are shared between a runner and its readers
  //template <typename Dtype>



  // A source is uniquely identified by its layer name + path, in case
  // the same database is read from two different locations in the net.
  static inline string source_key(const LayerParameter& param) {
    return param.name() + ":" + param.data_param().source();
  }
  const LayerParameter param_;
  const shared_ptr<QueuePair_Batch<Dtype> > queue_pair_;
  shared_ptr<Runner<Dtype> > runner_;
  shared_ptr<Data_provider<Dtype> > d_provider_;
  shared_ptr<Caffe::RNG> prefetch_rng_;
  unsigned int patch_count_;
  unsigned int patches_per_data_batch_;
  vector<int>  dest_label_shape_;
  vector<int>  dest_data_shape_;
  shared_ptr<DataTransformerND<Dtype> > data_transformer_nd;
  //PeekCropCenterPoint


  static map<const string, boost::weak_ptr<Runner<Dtype> > > runners_;

DISABLE_COPY_AND_ASSIGN(PatchSampler);
};

}  // namespace caffe

#endif  // CAFFE_DATA_READER_HPP_
