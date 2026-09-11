#include "io/hdf5_writer.hpp"
#include "numerics/derivatives.hpp"
#include <filesystem>
#include <stdexcept>

namespace {
void write_scalar_attr(H5::H5File& f,const std::string& name,double value){
    H5::DataSpace sp(H5S_SCALAR); auto a=f.createAttribute(name,H5::PredType::NATIVE_DOUBLE,sp); a.write(H5::PredType::NATIVE_DOUBLE,&value);
}
}

H5TrainingWriter::H5TrainingWriter(const std::string& filename,const Field& x,const Parameters& p):nx_(x.size()){
    std::filesystem::remove(filename);
    file_=std::make_unique<H5::H5File>(filename,H5F_ACC_TRUNC);
    hsize_t dims[1]={nx_}; H5::DataSpace xs(1,dims); auto xd=file_->createDataSet("/x",H5::PredType::NATIVE_DOUBLE,xs); xd.write(x.data(),H5::PredType::NATIVE_DOUBLE);
    create_1d_unlimited("/time");
    for(const char* n:{"/h","/q","/qh","/r","/h_x","/h_xx","/h_xxx","/h_t","/q_x","/q_xx","/q_t","/qh_x","/qh_xx"}) create_2d_unlimited(n);
    file_->createGroup("/forcing"); create_1d_unlimited("/forcing/F_inlet"); create_1d_unlimited("/forcing/q_inlet");
    write_attributes(p);
}
void H5TrainingWriter::create_1d_unlimited(const std::string& name){
    hsize_t dims[1]={0},maxd[1]={H5S_UNLIMITED},chunk[1]={100}; H5::DataSpace sp(1,dims,maxd); H5::DSetCreatPropList pl; pl.setChunk(1,chunk); file_->createDataSet(name,H5::PredType::NATIVE_DOUBLE,sp,pl);
}
void H5TrainingWriter::create_2d_unlimited(const std::string& name){
    hsize_t dims[2]={0,nx_},maxd[2]={H5S_UNLIMITED,nx_},chunk[2]={1,nx_}; H5::DataSpace sp(2,dims,maxd); H5::DSetCreatPropList pl; pl.setChunk(2,chunk); file_->createDataSet(name,H5::PredType::NATIVE_DOUBLE,sp,pl);
}
void H5TrainingWriter::append_scalar(const std::string& name,double value){
    auto ds=file_->openDataSet(name); hsize_t newd[1]={index_+1}; ds.extend(newd); auto fs=ds.getSpace(); hsize_t start[1]={index_},cnt[1]={1}; fs.selectHyperslab(H5S_SELECT_SET,cnt,start); H5::DataSpace ms(1,cnt); ds.write(&value,H5::PredType::NATIVE_DOUBLE,ms,fs);
}
void H5TrainingWriter::append_row(const std::string& name,const Field& row){
    if(row.size()!=nx_) throw std::runtime_error("HDF5 row size mismatch");
    auto ds=file_->openDataSet(name);
    hsize_t newd[2]={index_+1,nx_}; ds.extend(newd); auto fs=ds.getSpace();
    hsize_t start[2]={index_,0},cnt[2]={1,nx_};
    fs.selectHyperslab(H5S_SELECT_SET,cnt,start); H5::DataSpace ms(2,cnt);
    ds.write(row.data(),H5::PredType::NATIVE_DOUBLE,ms,fs);
}
void H5TrainingWriter::append(double t,const Field& h,const Field& q,const Field& r,const Field& dhdt,const Field& dqdt,double F,double qin){
    Field qh(nx_); for(std::size_t i=0;i<nx_;++i) qh[i]=q[i]/h[i];
    const double dx = nx_>1 ? 1.0 : 1.0; // placeholder overwritten below is not possible without storing dx
    // Derivatives use the actual x-spacing recovered from /x.
    auto xd=file_->openDataSet("/x"); Field x(nx_); xd.read(x.data(),H5::PredType::NATIVE_DOUBLE); const double dd=x[1]-x[0];
    auto hx=ddx_1d(h,dd), hxx=d2dx2_1d(h,dd), hxxx=d3dx3_1d(h,dd), qx=ddx_1d(q,dd), qxx=d2dx2_1d(q,dd), qhx=ddx_1d(qh,dd), qhxx=d2dx2_1d(qh,dd);
    (void)dx;
    append_scalar("/time",t); append_row("/h",h); append_row("/q",q); append_row("/qh",qh); append_row("/r",r);
    append_row("/h_x",hx); append_row("/h_xx",hxx); append_row("/h_xxx",hxxx); append_row("/h_t",dhdt);
    append_row("/q_x",qx); append_row("/q_xx",qxx); append_row("/q_t",dqdt); append_row("/qh_x",qhx); append_row("/qh_xx",qhxx);
    append_scalar("/forcing/F_inlet",F); append_scalar("/forcing/q_inlet",qin); ++index_;
}
void H5TrainingWriter::write_attributes(const Parameters& p){
    write_scalar_attr(*file_,"Re",p.Re); write_scalar_attr(*file_,"We",p.We); write_scalar_attr(*file_,"Ca",p.Ca); write_scalar_attr(*file_,"rho",p.in.rho); write_scalar_attr(*file_,"mu",p.in.mu); write_scalar_attr(*file_,"gamma",p.gamma); write_scalar_attr(*file_,"F2",p.F2); write_scalar_attr(*file_,"hN",p.hN); write_scalar_attr(*file_,"uN",p.uN); write_scalar_attr(*file_,"TN",p.TN); write_scalar_attr(*file_,"theta",p.theta); write_scalar_attr(*file_,"theta_deg",p.in.theta_deg); write_scalar_attr(*file_,"fcut_dim",p.in.fcut_dim); write_scalar_attr(*file_,"fcut_noise",p.fcut_noise); write_scalar_attr(*file_,"F0_noise",p.in.F0_noise); write_scalar_attr(*file_,"MNOISE",static_cast<double>(p.in.MNOISE));
}
