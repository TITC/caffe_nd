#include "caffe/data_provider.hpp"
namespace caffe {

template <typename Dtype>
Data_provider<Dtype>::Data_provider(const DataProviderParameter& param)
:param_(param)
{
  batch_size_ = param_.batch_size();
}

template <typename Dtype>
Data_DB_provider<Dtype>::Data_DB_provider(const DataProviderParameter& param)
:Data_provider<Dtype>(param){

}

template <typename Dtype>
 Data_HDF5_provider<Dtype>:: Data_HDF5_provider(const DataProviderParameter& param)
:Data_provider<Dtype>(param){
  this->source_data_label_pair_.resize(this->batch_size_);
  //hdf_blobs_.resize(batch_size_);
  // Read the source to parse the filenames.
  bool has_hdf5_source = this->param_.has_hdf5_source();
  CHECK(has_hdf5_source)<<"hdf5 data must have a source file ...";
  const string& hdf5_source = this->param_.hdf5_source();
  LOG(INFO) << "Loading list of HDF5 filenames from: " << hdf5_source;
  hdf5_filenames_.clear();
  std::ifstream source_file(hdf5_source.c_str());
  if (source_file.is_open()) {
    std::string line;
    while (source_file >> line) {
      hdf5_filenames_.push_back(line);
    }
  } else {
    LOG(FATAL) << "Failed to open source file: " << hdf5_source;
  }
  source_file.close();
  num_files_ = hdf5_filenames_.size();
  current_file_ = 0;
  LOG(INFO) << "Number of HDF5 files: " << num_files_;
  CHECK_GE(num_files_, 1) << "Must have at least 1 HDF5 filename listed in "
    << hdf5_source;

  file_permutation_.clear();
  file_permutation_.resize(num_files_);
  // Default to identity permutation.
  for (int i = 0; i < num_files_; i++) {
    file_permutation_[i] = i;
  }

  // Shuffle if needed.
  if (this->param_.hdf5_file_shuffle()) {
    std::random_shuffle(file_permutation_.begin(), file_permutation_.end());
  }

  // Load the first HDF5 file and initialize the line counter.
  this->LoadHDF5FileData(hdf5_filenames_[file_permutation_[current_file_]].c_str(),0);
  //current_row_ = 0;

}

template <typename Dtype>
void Data_HDF5_provider<Dtype>::loadHDF5FileData(const char* filename, int blob_idx)
{
  DLOG(INFO) << "Loading HDF5 file: " << filename;
  hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0) {
    LOG(FATAL) << "Failed opening HDF5 file: " << filename;
  }

  //int top_size = this->layer_param_.top_size();
  // we assume that each hdf5 file contains one "data" and one "label"
  Dtype a =Dtype(1);
  vector<string> data_set_names;
  data_set_names.push_back("data");
  data_set_names.push_back("label");
  int num_dataset =data_set_names.size();
  const int MIN_DATA_DIM = 1;
  const int MAX_DATA_DIM = INT_MAX;

  //for (int i = 0; i < data_set_names.size(); ++i) {
    //source_data_label_pair_[blob_idx].data = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());
    hdf5_load_nd_dataset(file_id, data_set_names[0].c_str(),
        MIN_DATA_DIM, MAX_DATA_DIM, &this->source_data_label_pair_[blob_idx].data);

    //source_data_label_pair_[blob_idx].label = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());
        hdf5_load_nd_dataset(file_id, data_set_names[1].c_str(),
            MIN_DATA_DIM, MAX_DATA_DIM, &this->source_data_label_pair_[blob_idx].label);


  //}

  herr_t status = H5Fclose(file_id);
  CHECK_GE(status, 0) << "Failed to close HDF5 file: " << filename;

  // MinTopBlobs==1 guarantees at least one top blob
  CHECK_GE(hdf_blobs_[0]->num_axes(), 1) << "Input must have at least 1 axis.";
  const int num = hdf_blobs_[0]->shape(0);
  for (int i = 1; i < num_dataset; ++i) {
    CHECK_EQ(hdf_blobs_[i]->shape(0), num);
  }
  // Default to identity permutation.
  data_permutation_.clear();
  data_permutation_.resize(hdf_blobs_[0]->shape(0));
  for (int i = 0; i < hdf_blobs_[0]->shape(0); i++)
    data_permutation_[i] = i;

  // Shuffle if needed.
  if (this->param_.data_shuffle()) {
    std::random_shuffle(data_permutation_.begin(), data_permutation_.end());
    DLOG(INFO) << "Successully loaded " << hdf_blobs_[0]->shape(0)
               << " rows (shuffled)";
  } else {
    DLOG(INFO) << "Successully loaded " << hdf_blobs_[0]->shape(0) << " rows";
  }
}

template <typename Dtype>
 void Data_HDF5_provider<Dtype>::load_next_batch(int numData){

   for(int i=0;i<numData;++i){
     if(current_file_ >=num_files_) {current_file_  =0;}
     //LoadHDF5FileData(hdf5_filenames_[file_permutation_[current_file_]].c_str(), current_file_);
     current_file_++;
    }
 }

 }
