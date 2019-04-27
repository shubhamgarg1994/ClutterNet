// //nvcc -std=c++11 compute_realtsdf_GPU.cu
// #include <algorithm>
// #include <vector>
// #include <cmath>
// #include <string>
// #include <cstdio>
// #include <iostream>
// #include <fstream>
// #include <stdio.h>
// #include <sys/time.h>
#include <time.h>
using namespace std;

// extern void FatalError(const int lineNumber);
// extern void checkCUDA(const int lineNumber, cudaError_t status);

// unsigned long long get_timestamp_dss(){
//   struct timeval now;
//   gettimeofday (&now, NULL);
//   return  now.tv_usec + (unsigned long long)now.tv_sec * 1000000;
// };

template <typename T>
string vecPrintString(vector<T> &v)
{
  stringstream convert;
  convert << "[" << v.size() << "]={";
  if (v.size()>0)  convert << v[0];
  if (v.size()>1){  
    for (int i = 1; i < v.size(); ++i){
      convert << "," << v[i];
    }
  }
  convert << "}"; //<<endl;
  string result = convert.str();
  return result;
}

struct Point3D {
  float x, y, z;
  Point3D(float xi, float yi, float zi): x(xi), y(yi), z(zi) {};
  Point3D(): x(0), y(0), z(0) {};
};

__global__ void compute_xyzkernel(float* XYZimage, float* depthMap, float * K, float * R){
  int iy = blockIdx.x;
  int ix = threadIdx.x;
  int width = blockDim.x;
 
  // printf("%d\n", width);
  int index = iy * width + ix;

  float depth = depthMap[ index ];

  if (depth < 0.00001)
  {
    XYZimage[3 * index + 0] = 0;
    XYZimage[3 * index + 1] = 0;
    XYZimage[3 * index + 2] = 0;
  }
  else
  {
    // project the depth point to 3d
    float tdx = (float(ix + 1) - K[2]) * depth / K[0];
    float tdz =  - (float(iy + 1) - K[5]) * depth / K[4];
    float tdy = depth;

    XYZimage[3 * index + 0] = R[0] * tdx + R[1] * tdy + R[2] * tdz;
    XYZimage[3 * index + 1] = R[3] * tdx + R[4] * tdy + R[5] * tdz;
    XYZimage[3 * index + 2] = R[6] * tdx + R[7] * tdy + R[8] * tdz;
    // XYZimage[3 * (iy*width + ix) + 0] = tdx;
    // XYZimage[3 * (iy*width + ix) + 1] = tdy;
    // XYZimage[3 * (iy*width + ix) + 2] = tdz;  
  }
  
  
}

__global__ void compute_TSDFGPUbox_proj(float* tsdf_data, float* R_data, float* K_data, float* depth, float* XYZimage,
                                      const float* scene_setting, int im_w, int im_h)
{
    const int index = threadIdx.x + blockIdx.x * blockDim.x;
    int tsdf_size[3];
    tsdf_size[0] = scene_setting[6];
    tsdf_size[1] = scene_setting[7];
    tsdf_size[2] = scene_setting[8];

    int volume_size = tsdf_size[0] * tsdf_size[1] * tsdf_size[2];
    if (index > volume_size) return;
    float delta_x = (scene_setting[1]-scene_setting[0]) / float(scene_setting[6]);  
    float delta_y = (scene_setting[3]-scene_setting[2]) / float(scene_setting[7]);   
    float delta_z = (scene_setting[5]-scene_setting[4]) / float(scene_setting[8]);

    float truncated_value = scene_setting[9];
    float reg_scale = 1.0/truncated_value/truncated_value;
    float num_channel = 3;


    float x = float((index / ( tsdf_size[1] * tsdf_size[2] ) ) % tsdf_size[0]) ;
    float y = float((index / tsdf_size[2]) % tsdf_size[1] );
    float z = float(index % tsdf_size[2]);

    // for (int i =0;i<num_channel;i++){
    //     tsdf_data[index + i * volume_size] = 0;
    // }

    // get grid world coordinate
    float temp_x = scene_setting[0] + (x + 0.5) * delta_x;
    float temp_y = scene_setting[2] + (y + 0.5) * delta_y;
    float temp_z = scene_setting[4] + (z + 0.5) * delta_z;

    x = temp_x;
    y = temp_y;
    z = temp_z;

    // x = temp_x * bb3d_data[0] + temp_y * bb3d_data[3] + temp_z * bb3d_data[6]
    //     + bb3d_data[9];
    // y = temp_x * bb3d_data[1] + temp_y * bb3d_data[4] + temp_z * bb3d_data[7]
    //     + bb3d_data[10];
    // z = temp_x * bb3d_data[2] + temp_y * bb3d_data[5] + temp_z * bb3d_data[8]
    //     + bb3d_data[11]; 

    // project to image plane decides the sign
    // rotate back and swap y, z and -y
    float xx =   R_data[0] * x + R_data[3] * y + R_data[6] * z;
    float zz =   R_data[1] * x + R_data[4] * y + R_data[7] * z;
    float yy = - R_data[2] * x - R_data[5] * y - R_data[8] * z;
    int ix = floor(xx * K_data[0] / zz + K_data[2]+0.5) - 1;
    int iy = floor(yy * K_data[4] / zz + K_data[5]+0.5) - 1;

    // 1. out of image case
    if (ix < 0 || ix >= im_w || iy < 0 || iy >= im_h || zz < 0.0001) {
      tsdf_data[index] = 0; // value
      tsdf_data[index + volume_size] = 0; // No depth
      tsdf_data[index + 2*volume_size] = -1; // Out of image
      // tsdf_data[index + 3*volume_size] = 0; // undefined front back
      return;  
    }
    // 2. missing depth case
    int img_index = iy * im_w + ix;
    float depth_onsite = depth[img_index];
    if ( depth_onsite < 0.0001 || depth_onsite > -0.0001 ) {
      tsdf_data[index] = 0; // value
      tsdf_data[index + volume_size] = 0; // No depth
      tsdf_data[index + 2*volume_size] = 1; // Inside image
      // tsdf_data[index + 3*volume_size] = 0; // undefined front back
      return;
    }

    // 3. normal case, some 3D points
    float x_project   = XYZimage[3*img_index+0];
    float y_project   = XYZimage[3*img_index+1];
    float z_project   = XYZimage[3*img_index+2]; 

    float tsdf_x = abs(x - x_project);
    float tsdf_y = abs(y - y_project);
    float tsdf_z = abs(z - z_project);

    float dist_to_surface_square = tsdf_x * tsdf_x + tsdf_y * tsdf_y + tsdf_z * tsdf_z;
    float tsdf_value = max(1 - reg_scale*dist_to_surface_square, 0.0);
    tsdf_data[index] = (zz > y_project) ? -tsdf_value:tsdf_value; // value
    tsdf_data[index + volume_size] = 1; // Has depth
    tsdf_data[index + 2*volume_size] = 1; // Inside image
}

__global__ void compute_TSDFGPU_grid_proj(float* tsdf_data, float* R_data, float* K_data, float* depth, float* XYZimage,
                                      const float* grid, int grid_size, float truncated_value, int im_w, int im_h)
{
    const int index = threadIdx.x + blockIdx.x * blockDim.x;
    int volume_size = grid_size;
    if (index >= volume_size) return;

    float reg_scale = 1.0/truncated_value/truncated_value;
    float num_channel = 3;

    // printf("%f\n", reg_scale);

    float x = grid[ index*3 ];
    float y = grid[ index*3 + 1 ];
    float z = grid[ index*3 + 2 ];

    // for (int i =0;i<num_channel;i++){
    //     tsdf_data[index + i * volume_size] = 0;
    // }

    // get grid world coordinate
    // float temp_x = scene_setting[0] + (x + 0.5) * delta_x;
    // float temp_y = scene_setting[2] + (y + 0.5) * delta_y;
    // float temp_z = scene_setting[4] + (z + 0.5) * delta_z;

    // x = temp_x;
    // y = temp_y;
    // z = temp_z;

    // x = temp_x * bb3d_data[0] + temp_y * bb3d_data[3] + temp_z * bb3d_data[6]
    //     + bb3d_data[9];
    // y = temp_x * bb3d_data[1] + temp_y * bb3d_data[4] + temp_z * bb3d_data[7]
    //     + bb3d_data[10];
    // z = temp_x * bb3d_data[2] + temp_y * bb3d_data[5] + temp_z * bb3d_data[8]
    //     + bb3d_data[11]; 

    // project to image plane decides the sign
    // rotate back and swap y, z and -y
    float xx =   R_data[0] * x + R_data[3] * y + R_data[6] * z;
    float zz =   R_data[1] * x + R_data[4] * y + R_data[7] * z;
    float yy = - R_data[2] * x - R_data[5] * y - R_data[8] * z;
    int ix = floor(xx * K_data[0] / zz + K_data[2]+0.5) - 1;
    int iy = floor(yy * K_data[4] / zz + K_data[5]+0.5) - 1;

    // 1. out of image case
    if (ix < 0 || ix >= im_w || iy < 0 || iy >= im_h || zz < 0.0001) {
      tsdf_data[index] = 0; // value
      tsdf_data[index + volume_size] = 0; // No depth
      tsdf_data[index + 2*volume_size] = -1; // Out of image
      // tsdf_data[index + 3*volume_size] = 0; // undefined front back
      return;  
    }
    // 2. missing depth case
    int img_index = iy * im_w + ix;
    float depth_onsite = depth[img_index];
    if ( depth_onsite < 0.0001 ) {
      tsdf_data[index] = 0; // value
      tsdf_data[index + volume_size] = 0; // No depth
      tsdf_data[index + 2*volume_size] = 1; // Inside image
      // tsdf_data[index + 3*volume_size] = 0; // undefined front back
      return;
    }
    // printf("%d, %d, %f\n", ix, iy, depth_onsite);
    // 3. normal case, some 3D points
    float x_project   = XYZimage[3*img_index+0];
    float y_project   = XYZimage[3*img_index+1];
    float z_project   = XYZimage[3*img_index+2]; 

    float tsdf_x = abs(x - x_project);
    float tsdf_y = abs(y - y_project);
    float tsdf_z = abs(z - z_project);

    float dist_to_surface_square = tsdf_x * tsdf_x + tsdf_y * tsdf_y + tsdf_z * tsdf_z;
    float tsdf_value = max(1 - reg_scale*dist_to_surface_square, 0.0);
    tsdf_data[index] = (zz > y_project) ? -tsdf_value:tsdf_value; // value
    tsdf_data[index + volume_size] = 1; // Has depth
    tsdf_data[index + 2*volume_size] = 1; // Inside image

    // if (index==1)
    // {
    //   printf("depth: %f\n", depth_onsite);
    //   printf("img_index: %d\n", img_index);
    //   printf("ix: %d, iy: %d\n", ix, iy); 
    //   printf("x_p: %f, y_p: %f, z_p: %f\n", x_project, y_project, z_project);
    //   printf("x: %f, y: %f, z: %f\n", x, y, z);
    // }
}

__global__ void compute_TSDFGPU_grid_proj_storageT(StorageT* tsdf_data, float* R_data, float* K_data, float* depth, float* XYZimage,
                                      const float* grid, int grid_size, float truncated_value, int im_w, int im_h)
{
    const int index = threadIdx.x + blockIdx.x * blockDim.x;
    int volume_size = grid_size;
    if (index >= volume_size) return;

    float reg_scale = 1.0/truncated_value/truncated_value;
    float num_channel = 3;

    // printf("%f\n", reg_scale);

    float x = grid[ index*3 ];
    float y = grid[ index*3 + 1 ];
    float z = grid[ index*3 + 2 ];

    // for (int i =0;i<num_channel;i++){
    //     tsdf_data[index + i * volume_size] = 0;
    // }

    // get grid world coordinate
    // float temp_x = scene_setting[0] + (x + 0.5) * delta_x;
    // float temp_y = scene_setting[2] + (y + 0.5) * delta_y;
    // float temp_z = scene_setting[4] + (z + 0.5) * delta_z;

    // x = temp_x;
    // y = temp_y;
    // z = temp_z;

    // x = temp_x * bb3d_data[0] + temp_y * bb3d_data[3] + temp_z * bb3d_data[6]
    //     + bb3d_data[9];
    // y = temp_x * bb3d_data[1] + temp_y * bb3d_data[4] + temp_z * bb3d_data[7]
    //     + bb3d_data[10];
    // z = temp_x * bb3d_data[2] + temp_y * bb3d_data[5] + temp_z * bb3d_data[8]
    //     + bb3d_data[11]; 

    // project to image plane decides the sign
    // rotate back and swap y, z and -y
    float xx =   R_data[0] * x + R_data[3] * y + R_data[6] * z;
    float zz =   R_data[1] * x + R_data[4] * y + R_data[7] * z;
    float yy = - R_data[2] * x - R_data[5] * y - R_data[8] * z;
    int ix = floor(xx * K_data[0] / zz + K_data[2]+0.5) - 1;
    int iy = floor(yy * K_data[4] / zz + K_data[5]+0.5) - 1;

    // 1. out of image case
    if (ix < 0 || ix >= im_w || iy < 0 || iy >= im_h || zz < 0.0001) {
      tsdf_data[index] = GPUCompute2StorageT( ComputeT(0) ); // value
      tsdf_data[index + volume_size] = GPUCompute2StorageT( ComputeT(0) ); // No depth
      tsdf_data[index + 2*volume_size] = GPUCompute2StorageT( ComputeT(-1) ); // Out of image
      // tsdf_data[index + 3*volume_size] = 0; // undefined front back
      return;  
    }
    // 2. missing depth case
    int img_index = iy * im_w + ix;
    float depth_onsite = depth[img_index];
    if ( depth_onsite < 0.0001 ) {
      tsdf_data[index] = GPUCompute2StorageT( ComputeT(0) ); // value
      tsdf_data[index + volume_size] = GPUCompute2StorageT( ComputeT(0) ); // No depth
      tsdf_data[index + 2*volume_size] = GPUCompute2StorageT( ComputeT(1) ); // Inside image
      // tsdf_data[index + 3*volume_size] = 0; // undefined front back
      return;
    }
    // printf("%d, %d, %f\n", ix, iy, depth_onsite);
    // 3. normal case, some 3D points
    float x_project   = XYZimage[3*img_index+0];
    float y_project   = XYZimage[3*img_index+1];
    float z_project   = XYZimage[3*img_index+2]; 

    float tsdf_x = abs(x - x_project);
    float tsdf_y = abs(y - y_project);
    float tsdf_z = abs(z - z_project);

    float dist_to_surface_square = tsdf_x * tsdf_x + tsdf_y * tsdf_y + tsdf_z * tsdf_z;
    float tsdf_value = max(1 - reg_scale*dist_to_surface_square, 0.0);
    float temp = (zz > y_project) ? -tsdf_value:tsdf_value;
    tsdf_data[index] =  GPUCompute2StorageT( ComputeT(temp) );// value
    tsdf_data[index + volume_size] = GPUCompute2StorageT( ComputeT(1) ); // Has depth
    tsdf_data[index + 2*volume_size] = GPUCompute2StorageT( ComputeT(1) ); // Inside image

    // if (index==1)
    // {
    //   printf("depth: %f\n", depth_onsite);
    //   printf("img_index: %d\n", img_index);
    //   printf("ix: %d, iy: %d\n", ix, iy); 
    //   printf("x_p: %f, y_p: %f, z_p: %f\n", x_project, y_project, z_project);
    //   printf("x: %f, y: %f, z: %f\n", x, y, z);
    // }
}

__global__ void compute_grid(float* transform_grid_gpu, float* grid_gpu, float* Rot_GPU, float* Tsl_GPU, int grid_size)
{
  const int index = threadIdx.x + blockIdx.x * blockDim.x;
  
  
  // printf("%d: %d %d %d\n", index, threadIdx.x, blockIdx.x, blockDim.x);

  if (index >= grid_size) return;

  int xid = 3 * index;
  // printf("grid: %d, index: %d, xid: %d\n", grid_size, index, xid);
  // printf("R: %f %f %f %f, T: %f %f %f;", Rot_GPU[0], Rot_GPU[1], Rot_GPU[2], Rot_GPU[3], Tsl_GPU[0], Tsl_GPU[1], Tsl_GPU[2]);
  float x = grid_gpu[xid];
  float y = grid_gpu[xid + 1];
  float z = grid_gpu[xid + 2];

  transform_grid_gpu[xid] = Rot_GPU[0]*x + Rot_GPU[1]*y - Tsl_GPU[0];
  transform_grid_gpu[xid+1] = Rot_GPU[2]*x + Rot_GPU[3]*y - Tsl_GPU[1];
  transform_grid_gpu[xid+2] = z - Tsl_GPU[2];

  // printf(" %d: %f %f %f, %f %f %f\n", index, x, y, z, transform_grid_gpu[xid], transform_grid_gpu[xid+1], transform_grid_gpu[xid+2]);
}

struct Box3D{
  unsigned int category;
  float orientation[3];
  float center[3];
  float coeff[3];

  Box3D(): category(-1) {};

  Box3D(const Box3D &b)
  {
    category = b.category;
    orientation[0] = b.orientation[0];
    orientation[1] = b.orientation[1];
    orientation[2] = b.orientation[2];
    center[0] = b.center[0];
    center[1] = b.center[1];
    center[2] = b.center[2];
    coeff[0] = b.coeff[0];
    coeff[1] = b.coeff[1];
    coeff[2] = b.coeff[2];
  }

  void transformBox3D(float* Rot, float* Tsl)
  {
    float ox = orientation[0];
    float oy = orientation[1];
    orientation[0] = Rot[0]*ox + Rot[2]*oy;
    orientation[1] = Rot[1]*ox + Rot[3]*oy;

    // float cx = center[0];
    // float cy = center[1];
    // float cz = center[2];
    // center[0] = Rot[0]*cx + Rot[2]*cy + Tsl[0];
    // center[1] = Rot[1]*cx + Rot[3]*cy + Tsl[1];
    // center[2] = cz + Tsl[2];
    float cx = center[0] + Tsl[0];
    float cy = center[1] + Tsl[1];
    float cz = center[2] + Tsl[2];
    center[0] = Rot[0]*cx + Rot[2]*cy;
    center[1] = Rot[1]*cx + Rot[3]*cy;
    center[2] = cz;
  }

  void load(FILE* &fp)
  {
    fread((void*)(&category), sizeof(unsigned int), 1, fp);
    // cout << "category: " << category << endl;
    fread((void*)center, sizeof(unsigned int), 3, fp);
    // cout << "center: " << center[0] << " " << center[1] << " " << center[2] << endl;
    fread((void*)coeff, sizeof(unsigned int), 3, fp);
    // cout << "coeff: " << coeff[0] << " " << coeff[1] << " " << coeff[2] << endl;
    fread((void*)orientation, sizeof(unsigned int), 3, fp);
    // cout << "orientation: " << orientation[0] << " " << orientation[1] << " " << orientation[2] << endl;
  }

  void floorRectangle( vector<float> &rec)
  {
    rec.assign(8,0);
    float axis_x[2];
    float axis_y[2];

    axis_y[0] = orientation[0];
    axis_y[1] = orientation[1];
    axis_x[0] = axis_y[1];
    axis_x[1] = -axis_y[0];

    rec[0] = center[0] - axis_x[0]*coeff[0] - axis_y[0]*coeff[1];
    rec[1] = center[0] - axis_x[0]*coeff[0] + axis_y[0]*coeff[1];
    rec[2] = center[0] + axis_x[0]*coeff[0] + axis_y[0]*coeff[1];
    rec[3] = center[0] + axis_x[0]*coeff[0] - axis_y[0]*coeff[1];
    rec[4] = center[1] - axis_x[1]*coeff[0] - axis_y[1]*coeff[1];
    rec[5] = center[1] - axis_x[1]*coeff[0] + axis_y[1]*coeff[1];
    rec[6] = center[1] + axis_x[1]*coeff[0] + axis_y[1]*coeff[1];
    rec[7] = center[1] + axis_x[1]*coeff[0] - axis_y[1]*coeff[1];
  }

  void floorRectangle_MW( float *rec )
  {
    float axis_x[2];
    float axis_y[2];
    if ( abs(orientation[0]) < abs(orientation[1]))
    {
      // 2nd coeff is y
      rec[0] = center[2] - coeff[2];
      rec[1] = center[2] + coeff[2];
      rec[2] = center[1] - coeff[1];
      rec[3] = center[1] + coeff[1];
      rec[4] = center[0] - coeff[0];
      rec[5] = center[0] + coeff[0];
    }
    else
    {
      // 2nd coeff is x
      rec[0] = center[2] - coeff[2];
      rec[1] = center[2] + coeff[2];
      rec[2] = center[1] - coeff[0];
      rec[3] = center[1] + coeff[0];
      rec[4] = center[0] - coeff[1];
      rec[5] = center[0] + coeff[1];
    }
  }

  float IOU_MW(Box3D &box)
  {
    float rec1[6];
    float rec2[6];
    floorRectangle_MW(rec1);
    box.floorRectangle_MW(rec2);

    float intersection;
    float x_overlap = min(rec1[5], rec2[5]) - max(rec1[4], rec2[4]);
    float y_overlap = min(rec1[3], rec2[3]) - max(rec1[2], rec2[2]);
    float z_overlap = min(rec1[1], rec2[1]) - max(rec1[0], rec2[0]);
    if ( x_overlap<0 || y_overlap<0 || z_overlap<0)
    {
      intersection = 0;
    }
    else
    {
      intersection = x_overlap * y_overlap * z_overlap;
    }

    float vol1 = (rec1[5]-rec1[4]) * (rec1[3] - rec1[2]) * (rec1[1]-rec1[0]);
    float vol2 = (rec2[5]-rec2[4]) * (rec2[3] - rec2[2]) * (rec2[1]-rec2[0]);


    // std::cout << rec1[0] << " " << rec1[1] << " " << rec1[2] << " " << rec1[3] << " " << rec1[4] << " " << rec1[5] << std::endl;
    // std::cout << rec2[0] << " " << rec2[1] << " " << rec2[2] << " " << rec2[3] << " " << rec2[4] << " " << rec2[5] << std::endl;
    // std::cout << intersection << " " << vol1 << " " << vol2 << " " << intersection / ( vol1 + vol2 - intersection ) << std::endl;

    return intersection / ( vol1 + vol2 - intersection );
  }

  float IOU(Box3D &box)
  {
    vector<float> rec1, rec2;
    floorRectangle(rec1);
    box.floorRectangle(rec2);

    std::sort(rec1.begin(), rec1.begin()+4);
    std::sort(rec1.begin()+4, rec1.begin()+8);
    std::sort(rec2.begin(), rec2.begin()+4);
    std::sort(rec2.begin()+4, rec2.begin()+8);

    float min_x_1 = rec1[0];
    float max_x_1 = rec1[3];
    float min_y_1 = rec1[4];
    float max_y_1 = rec1[7];
    float min_x_2 = rec2[0];
    float max_x_2 = rec2[3];
    float min_y_2 = rec2[4];
    float max_y_2 = rec2[7];

    float intersection;
    float x_overlap = min(max_x_1, max_x_2) - max(min_x_1,min_x_2);
    float y_overlap = min(max_y_1, max_y_2) - max(min_y_1,min_y_2);
    if ( x_overlap<0 || y_overlap<0 )
    {
      intersection = 0;
    }
    else
    {
      intersection = x_overlap * y_overlap;
    }

    float vol1 = (max_x_1 - min_x_1) * (max_y_1 - min_y_1);
    float vol2 = (max_x_2 - min_x_2) * (max_y_2 - min_y_2);
    return intersection / ( vol1 + vol2 - intersection );
  }
};

// enum WallType { isa_floor; isa_ceiling; isa_wall;}

struct Wall3D {
  unsigned int type;
  float top_loc[2];
  // vector<float> points;
  // float* points;
  // unsigned int num_points;
  Wall3D() {
    type = 0;
    // points.clear();
    // num_points = 0;
  }

  Wall3D(const Wall3D &w)
  {
    type = w.type;
    top_loc[0] = w.top_loc[0];
    top_loc[1] = w.top_loc[1];
  }

  void transformWall3D(float* Rot, float* Tsl)
  {
    if (type==1 || type==2)
    {
      top_loc[0] = top_loc[0] + Tsl[2];
    }
    else if (type==0)
    {
      // float p1[2];
      // float p2[2];
      // p1[0] = top_loc[0];
      // p1[1] = top_loc[1];
      // p2[0] = top_loc[0] + top_loc[1];
      // p2[1] = top_loc[1] - top_loc[0];

      // float tp1[2];
      // float tp2[2];
      // tp1[0] = Rot[0]*p1[0] + Rot[2]*p1[1] + Tsl[0];
      // tp1[1] = Rot[1]*p1[0] + Rot[3]*p1[1] + Tsl[1];
      // tp2[0] = Rot[0]*p2[0] + Rot[2]*p2[1] + Tsl[0];
      // tp2[1] = Rot[1]*p2[0] + Rot[3]*p2[1] + Tsl[1];

      // cout << "top_loc: " << top_loc[0] << " " << top_loc[1] << endl;
      float p1[2];
      float p2[2];
      p1[0] = top_loc[0] + Tsl[0];
      p1[1] = top_loc[1] + Tsl[1];
      p2[0] = top_loc[0] + top_loc[1] + Tsl[0];
      p2[1] = top_loc[1] - top_loc[0] + Tsl[1];

      // cout << "Tsl: " << Tsl[0] << " " << Tsl[1] << " " << Tsl[2] << endl;
      // cout << "p1: " << p1[0] << " " << p1[1] << endl;
      // cout << "p2: " << p2[0] << " " << p2[1] << endl;

      float tp1[2];
      float tp2[2];
      tp1[0] = Rot[0]*p1[0] + Rot[2]*p1[1];
      tp1[1] = Rot[1]*p1[0] + Rot[3]*p1[1];
      tp2[0] = Rot[0]*p2[0] + Rot[2]*p2[1];
      tp2[1] = Rot[1]*p2[0] + Rot[3]*p2[1];

      float k_line[2];
      k_line[0] = tp2[1] - tp1[1];
      k_line[1] = tp1[0] - tp2[0];
      float temp = sqrt(k_line[0]*k_line[0] + k_line[1]*k_line[1]);
      k_line[0] /= temp;
      k_line[1] /= temp;
      float dist = tp1[0]*k_line[0] + tp1[1]*k_line[1];
      top_loc[0] = dist * k_line[0];
      top_loc[1] = dist * k_line[1]; 
      // cout << "top_loc: " << top_loc[0] << " " << top_loc[1] << endl;
    }
  }

  void load(FILE* &fp)
  {
    fread((void*)(&(type)), sizeof(unsigned int), 1, fp);
    // cout << "type: " << type << endl;
    fread((void*)top_loc, sizeof(float), 2, fp);
    // cout << "top_loc: " << top_loc[0] << ", " << top_loc[1] << endl;
  }

  // void load(FILE* &fp)
  // {
  //   fread((void*)(&(type)), sizeof(unsigned int), 1, fp);
  //   cout << "type: " << type << endl;
  //   fread((void*)(&(num_points)), sizeof(unsigned int), 1, fp);
  //   cout << "num_points: " << num_points << endl;
  //   float *p = new float[num_points*3];
  //   fread((void*)p, sizeof(float), num_points*3, fp);
  //   for (int i = 0; i<num_points*3; i++)
  //   {
  //     points.push_back(p[i]);
  //     cout << points[i] << " ";
  //   }
  //   cout << endl;
  //   delete[] p;
  // }
  ~Wall3D()
  {
  }

};

struct DepthImage {
  float* depth_cpu;
  // float* scene_setting_cpu;
  // float* K_CPU;
  // float* R_CPU;
  float* K_GPU;
  float* R_GPU;
  float* Rot_GPU;
  float* Tsl_GPU;
  unsigned int width;
  unsigned int height;
  unsigned int memorysize;

  float* tsdf_gpu;
  StorageT* tsdf_gpu_storage;

  float* scene_setting_gpu;
  float* depth_gpu;
  float* XYZimage_gpu;

  float* regular_grid_gpu;
  float* transform_grid_gpu;
  int grid_size;
  int max_grid_size;
  float truncated_value;

  DepthImage(): K_GPU(NULL), R_GPU(NULL), Rot_GPU(NULL), Tsl_GPU(NULL), depth_cpu(NULL), width(0), height(0), memorysize(0), tsdf_gpu(NULL), scene_setting_gpu(NULL), depth_gpu(NULL), XYZimage_gpu(NULL), regular_grid_gpu(NULL), transform_grid_gpu(NULL), max_grid_size(0), tsdf_gpu_storage(NULL){
    // K_CPU = new float[9];
    // R_CPU = new float[9];
    // checkCUDA(__LINE__, cudaMalloc(&K_GPU, sizeof(float)*9));
    // checkCUDA(__LINE__, cudaMalloc(&R_GPU, sizeof(float)*9));
    // checkCUDA(__LINE__, cudaMalloc(&Rot_GPU, sizeof(float)*4));
    // checkCUDA(__LINE__, cudaMalloc(&Tsl_GPU, sizeof(float)*3));
  };

  void reallocateMemory()
  {
    if (memorysize < height*width)
    {
      // cout << depth_gpu << " " << depth_cpu << endl;
      if (depth_cpu != NULL)  delete[] depth_cpu;
      depth_cpu = new float[ height * width ];
      if (depth_gpu != NULL)  checkCUDA(__LINE__, cudaFree(depth_gpu)); 
      checkCUDA(__LINE__, cudaMalloc(&depth_gpu, sizeof(float)*width*height));
      if (XYZimage_gpu != NULL)  checkCUDA(__LINE__, cudaFree(XYZimage_gpu)); 
      checkCUDA(__LINE__, cudaMalloc(&XYZimage_gpu, sizeof(float)*width*height*3));
      memorysize = height * width;
      // cout << "Reallocate to: " << height << " * " << width << endl;
      // cout << depth_gpu << " " << depth_cpu << endl;
    }
  }

  void SceneSetting( float* p)
  {
    // if (scene_setting_cpu != NULL) scene_setting_cpu = new float[10];
    if (scene_setting_gpu == NULL) checkCUDA(__LINE__, cudaMalloc(&scene_setting_gpu, sizeof(float)*10));
    checkCUDA(__LINE__, cudaMemcpy(scene_setting_gpu, p, sizeof(float)*10, cudaMemcpyHostToDevice)); 

    int x_size = std::round(p[6]);
    int y_size = std::round(p[7]);
    int z_size = std::round(p[8]);
    float x_start = p[0];
    float x_end = p[1];
    float y_start = p[2];
    float y_end = p[3];
    float z_start = p[4];
    float z_end = p[5];
    float x_delta = (x_end-x_start)/x_size;
    float y_delta = (y_end-y_start)/y_size;
    float z_delta = (z_end-z_start)/z_size;

    grid_size = x_size * y_size * z_size;
    cout << "x_delta: " << x_delta << endl;
    cout << "y_delta: " << y_delta << endl;
    cout << "z_delta: " << z_delta << endl;
    truncated_value = p[9];


    if (grid_size > max_grid_size)
    {
      // cout << "Allocate memory..." << endl;
      if (tsdf_gpu != NULL) checkCUDA( __LINE__, cudaFree(tsdf_gpu));
      checkCUDA(__LINE__, cudaMalloc( &tsdf_gpu, sizeof(float) * grid_size * 3));
      if (tsdf_gpu_storage != NULL) checkCUDA( __LINE__, cudaFree(tsdf_gpu_storage));
      checkCUDA(__LINE__, cudaMalloc( &tsdf_gpu_storage, sizeofStorageT * grid_size * 3));
      if (regular_grid_gpu != NULL) checkCUDA( __LINE__, cudaFree(regular_grid_gpu));
      checkCUDA(__LINE__, cudaMalloc( &regular_grid_gpu, sizeof(float) * grid_size * 3));
      if (transform_grid_gpu != NULL) checkCUDA( __LINE__, cudaFree(transform_grid_gpu));
      checkCUDA(__LINE__, cudaMalloc( &transform_grid_gpu, sizeof(float) * grid_size * 3));
      max_grid_size = grid_size;
      // cout << "max_grid_size: " << max_grid_size << endl;

      // cout << "SceneSetting: " << endl;
      // cout << "tsdf_gpu: " << tsdf_gpu << endl;
      // cout << "regular_grid_gpu: " << regular_grid_gpu << endl;
      // cout << "transform_grid_gpu: " << transform_grid_gpu << endl;
    }  

    float* grid_cpu = new float[grid_size*3];
    int count = 0;
    for (int z = 0; z < z_size; z++)
      for (int y = 0; y < y_size; y++)
        for (int x = 0; x < x_size; x++)
        {
          grid_cpu[count*3+0] = x_start + ((float)x+0.5) * x_delta;
          grid_cpu[count*3+1] = y_start + ((float)y+0.5) * y_delta;
          grid_cpu[count*3+2] = z_start + ((float)z+0.5) * z_delta;
          count++;
        }  
    // cout << "Count: " << count << endl;
    checkCUDA(__LINE__, cudaMemcpy(regular_grid_gpu, grid_cpu, sizeof(float)*grid_size*3, cudaMemcpyHostToDevice)); 
    delete[] grid_cpu;
  }

  void transformGridPoints()
  {
    // checkCUDA(__LINE__, cudaMemcpy(depth_gpu, depth_cpu, sizeof(float)*width*height, cudaMemcpyHostToDevice)); 

    int THREADS_NUM = 1024;
    int BLOCK_NUM = int((grid_size + size_t(THREADS_NUM) - 1) / THREADS_NUM);
    // cout << "BLOCK_NUM: " << BLOCK_NUM << ", THREADS_NUM: " << THREADS_NUM << endl;

    // cout << "grid_size: " << grid_size << endl;
    // float* p = new float[grid_size*3];   
    // FILE* fp = fopen("debug1.bin", "wb");

    // checkCUDA(__LINE__, cudaMemcpy(p, Rot_GPU, sizeof(float)*4, cudaMemcpyDeviceToHost));
    // fwrite(p, sizeof(float), 4, fp);
    // checkCUDA(__LINE__, cudaMemcpy(p, Tsl_GPU, sizeof(float)*3, cudaMemcpyDeviceToHost));
    // fwrite(p, sizeof(float), 3, fp);

    // checkCUDA(__LINE__, cudaMemcpy(p, regular_grid_gpu, sizeof(float)*grid_size*3, cudaMemcpyDeviceToHost));
    // fwrite(p, sizeof(float), grid_size*3, fp);
    // cout << "Start gpu function" << endl;

    // cout << "transfore: " << transform_grid_gpu << ", regular: " << regular_grid_gpu << endl;
    // cout << "Rot_GPU: " << Rot_GPU << ", Tsl_GPU: " << Tsl_GPU << endl;
    // checkCUDA(__LINE__,cudaDeviceSynchronize());
    // if (regular_grid_gpu != NULL) checkCUDA(__LINE__, cudaFree(regular_grid_gpu));
    // checkCUDA(__LINE__, cudaMalloc(&regular_grid_gpu, sizeof(float)*grid_size*3));
    // if (transform_grid_gpu != NULL) checkCUDA(__LINE__, cudaFree(transform_grid_gpu));
    // checkCUDA(__LINE__, cudaMalloc(&transform_grid_gpu, sizeof(float)*grid_size*3));
    // if (Rot_GPU != NULL) checkCUDA(__LINE__, cudaFree(Rot_GPU));
    // checkCUDA(__LINE__, cudaMalloc(&Rot_GPU, sizeof(float)*4));
    // if (Tsl_GPU != NULL) checkCUDA(__LINE__, cudaFree(Tsl_GPU));
    // checkCUDA(__LINE__, cudaMalloc(&Tsl_GPU, sizeof(float)*3));
    // cout << "new new new" << endl;
    // cout << "transfore: " << transform_grid_gpu << ", regular: " << regular_grid_gpu << endl;
    // cout << "Rot_GPU: " << Rot_GPU << ", Tsl_GPU: " << Tsl_GPU << endl;

    // checkCUDA(__LINE__, cudaMemcpy(p, regular_grid_gpu, sizeof(float)*grid_size*3, cudaMemcpyDeviceToHost));
    // fwrite(p, sizeof(float), grid_size*3, fp);

    // checkCUDA(__LINE__,cudaDeviceSynchronize());
    compute_grid<<<BLOCK_NUM,THREADS_NUM>>>(transform_grid_gpu, regular_grid_gpu, Rot_GPU, Tsl_GPU, grid_size);
    // checkCUDA(__LINE__, cudaMemcpy(transform_grid_gpu, regular_grid_gpu, sizeof(float)*grid_size*3, cudaMemcpyDeviceToDevice));
    // checkCUDA(__LINE__,cudaDeviceSynchronize());

    // cout << "Copy memory back to cpu" << endl;
    // checkCUDA(__LINE__, cudaMemcpy(p, transform_grid_gpu, sizeof(float)*grid_size*3, cudaMemcpyDeviceToHost));


    // cout << "Writing to file" << endl;
    // fwrite(p, sizeof(float), grid_size*3, fp);
    // delete[] p;
    // fclose(fp);

    // checkCUDA(__LINE__, cudaMemcpy(depth_gpu, depth_cpu, sizeof(float)*width*height, cudaMemcpyHostToDevice)); 
  }

  void readImage(string &filename)
  {
    // cout << "Reading: " << filename << endl;
      FILE* fp = fopen(filename.c_str(), "rb");
      // if ( fp == NULL)
      // {
      //   cerr<<"Fail to read properly"<<endl;
      //   FatalError(__LINE__);
      // }
      int iBuff;
      iBuff = fread(&height, sizeof(unsigned int), 1, fp);
      // cout << "height: " << height << endl;
      // cout << "iBuff: " << iBuff << endl;
      if (iBuff != 1) 
      {
        cerr<<"Fail to read properly"<<endl;
        FatalError(__LINE__);
      }
      iBuff = fread(&width, sizeof(unsigned int), 1, fp);
      // cout << "width: " << width << endl;
      // cout << "iBuff: " << iBuff << endl;
      if (iBuff != 1) 
      {
        cerr<<"Fail to read properly"<<endl;
        FatalError(__LINE__);
      }

      reallocateMemory();

      iBuff = fread( (void*)depth_cpu, sizeof(float), width*height, fp);
      // cout << "iBuff: " << iBuff << endl;
      if (iBuff != width*height) 
      {
        cerr<<"Fail to read properly"<<endl;
        FatalError(__LINE__);
      }
      fclose(fp);

      checkCUDA(__LINE__, cudaMemcpy(depth_gpu, depth_cpu, sizeof(float)*width*height, cudaMemcpyHostToDevice)); 
  }

  void SetupMatrix( float *K, float *R, float *Rot, float *Tsl)
  {
    // checkCUDA(__LINE__, cudaMemcpy(depth_gpu, depth_cpu, sizeof(float)*width*height, cudaMemcpyHostToDevice)); 

    // memcpy(K_CPU, K, sizeof(float)*9);
    // memcpy(R_CPU, R, sizeof(float)*9);
    if (K_GPU == NULL) checkCUDA(__LINE__, cudaMalloc(&K_GPU, sizeof(float)*9));    
    if (R_GPU == NULL) checkCUDA(__LINE__, cudaMalloc(&R_GPU, sizeof(float)*9));
    if (Rot_GPU == NULL) checkCUDA(__LINE__, cudaMalloc(&Rot_GPU, sizeof(float)*4));
    if (Tsl_GPU == NULL) checkCUDA(__LINE__, cudaMalloc(&Tsl_GPU, sizeof(float)*3));

    checkCUDA(__LINE__, cudaMemcpy(K_GPU, (float*)K, sizeof(float)*9, cudaMemcpyHostToDevice));  
    checkCUDA(__LINE__, cudaMemcpy(R_GPU, (float*)R, sizeof(float)*9, cudaMemcpyHostToDevice)); 
    checkCUDA(__LINE__, cudaMemcpy(Rot_GPU, (float*)Rot, sizeof(float)*4, cudaMemcpyHostToDevice));  
    checkCUDA(__LINE__, cudaMemcpy(Tsl_GPU, (float*)Tsl, sizeof(float)*3, cudaMemcpyHostToDevice)); 

    // cout << "SetupMatrix:" << endl;
    // cout << "K_GPU: " << K_GPU << endl;
    // cout << "R_GPU: " << R_GPU << endl;
    // cout << "Rot_GPU: " << Rot_GPU << endl;
    // cout << "Tsl_GPU: " << Tsl_GPU << endl;
  }

  void ComputeXYZimage()
  {
    if (XYZimage_gpu!=NULL) checkCUDA(__LINE__, cudaFree(XYZimage_gpu));
    checkCUDA(__LINE__, cudaMalloc(&XYZimage_gpu, sizeof(float)*width*height*3));
    compute_xyzkernel<<<height,width>>>(XYZimage_gpu,depth_gpu,K_GPU,R_GPU);
  }

  void ComputeTSDF()
  {
    // checkCUDA(__LINE__, cudaMemcpy(depth_gpu, depth_cpu, sizeof(float)*width*height, cudaMemcpyHostToDevice)); 
    // cout << "width: " << width << ", height: " << height << endl;
    // cout << "memorysize: " << memorysize << endl;
    // reallocateMemory();

    // cout << depth_gpu << " " << depth_cpu << endl;
    // cout << "Compute points: " << endl;
    // checkCUDA(__LINE__, cudaMemcpy(depth_gpu, depth_cpu, sizeof(float)*width*height, cudaMemcpyHostToDevice)); 

    // checkCUDA(__LINE__,cudaDeviceSynchronize());
    compute_xyzkernel<<<height,width>>>(XYZimage_gpu,depth_gpu,K_GPU,R_GPU);
    // checkCUDA(__LINE__,cudaDeviceSynchronize());

    // float* p = new float[3*width*height];
    // FILE* fp = fopen("debug2.bin", "wb");
    // checkCUDA(__LINE__, cudaMemcpy(p, K_GPU, sizeof(float)*9, cudaMemcpyDeviceToHost));
    // fwrite(p, sizeof(float), 9, fp);
    // checkCUDA(__LINE__, cudaMemcpy(p, R_GPU, sizeof(float)*9, cudaMemcpyDeviceToHost));
    // fwrite(p, sizeof(float), 9, fp);
    // checkCUDA(__LINE__, cudaMemcpy(p, XYZimage_gpu, sizeof(float)*3*width*height, cudaMemcpyDeviceToHost));
    // fwrite(&width, sizeof(unsigned int), 1, fp);
    // fwrite(&height, sizeof(unsigned int), 1, fp);
    // fwrite(p, sizeof(float), 3*width*height, fp);

    // checkCUDA(__LINE__, cudaMemcpy(p, depth_gpu, sizeof(float)*width*height, cudaMemcpyDeviceToHost));
    // fwrite(p, sizeof(float), width*height, fp);
    


    // cout << "TSDF: " << endl;
    int THREADS_NUM = 1024;
    int BLOCK_NUM = int((grid_size + size_t(THREADS_NUM) - 1) / THREADS_NUM);
    // compute_TSDFGPUbox_proj<<<BLOCK_NUM,THREADS_NUM>>>(tsdf_gpu, R_GPU, K_GPU, depth_gpu, XYZimage_gpu,
                                      // scene_setting_gpu, width, height);

    compute_TSDFGPU_grid_proj<<<BLOCK_NUM,THREADS_NUM>>>(tsdf_gpu, R_GPU, K_GPU, depth_gpu, XYZimage_gpu,
                                      transform_grid_gpu, grid_size, truncated_value, width, height);

    // compute_TSDFGPU_grid_proj_storageT<<<BLOCK_NUM,THREADS_NUM>>>(tsdf_gpu_storage, R_GPU, K_GPU, depth_gpu, XYZimage_gpu,
    //                                   transform_grid_gpu, grid_size, truncated_value, width, height);
    
    // delete[] p;
    // p = new float[grid_size*3];
    // checkCUDA(__LINE__, cudaMemcpy(p, tsdf_gpu, sizeof(float)*grid_size*3, cudaMemcpyDeviceToHost));
    // fwrite(p, sizeof(float), grid_size*3, fp);
    // fclose(fp);
    // delete[] p;
  }

  void ComputeTSDF_storage()
  {
    // checkCUDA(__LINE__, cudaMemcpy(depth_gpu, depth_cpu, sizeof(float)*width*height, cudaMemcpyHostToDevice)); 
    // cout << "width: " << width << ", height: " << height << endl;
    // cout << "memorysize: " << memorysize << endl;
    // reallocateMemory();

    // cout << depth_gpu << " " << depth_cpu << endl;
    // cout << "Compute points: " << endl;
    // checkCUDA(__LINE__, cudaMemcpy(depth_gpu, depth_cpu, sizeof(float)*width*height, cudaMemcpyHostToDevice)); 

    // checkCUDA(__LINE__,cudaDeviceSynchronize());
    compute_xyzkernel<<<height,width>>>(XYZimage_gpu,depth_gpu,K_GPU,R_GPU);
    // checkCUDA(__LINE__,cudaDeviceSynchronize());

    // float* p = new float[3*width*height];
    // FILE* fp = fopen("debug2.bin", "wb");
    // checkCUDA(__LINE__, cudaMemcpy(p, K_GPU, sizeof(float)*9, cudaMemcpyDeviceToHost));
    // fwrite(p, sizeof(float), 9, fp);
    // checkCUDA(__LINE__, cudaMemcpy(p, R_GPU, sizeof(float)*9, cudaMemcpyDeviceToHost));
    // fwrite(p, sizeof(float), 9, fp);
    // checkCUDA(__LINE__, cudaMemcpy(p, XYZimage_gpu, sizeof(float)*3*width*height, cudaMemcpyDeviceToHost));
    // fwrite(&width, sizeof(unsigned int), 1, fp);
    // fwrite(&height, sizeof(unsigned int), 1, fp);
    // fwrite(p, sizeof(float), 3*width*height, fp);

    // checkCUDA(__LINE__, cudaMemcpy(p, depth_gpu, sizeof(float)*width*height, cudaMemcpyDeviceToHost));
    // fwrite(p, sizeof(float), width*height, fp);
    


    // cout << "TSDF: " << endl;
    int THREADS_NUM = 1024;
    int BLOCK_NUM = int((grid_size + size_t(THREADS_NUM) - 1) / THREADS_NUM);
    // compute_TSDFGPUbox_proj<<<BLOCK_NUM,THREADS_NUM>>>(tsdf_gpu, R_GPU, K_GPU, depth_gpu, XYZimage_gpu,
    //                                   scene_setting_gpu, width, height);

    // compute_TSDFGPU_grid_proj<<<BLOCK_NUM,THREADS_NUM>>>(tsdf_gpu, R_GPU, K_GPU, depth_gpu, XYZimage_gpu,
                                      // transform_grid_gpu, grid_size, truncated_value, width, height);


    compute_TSDFGPU_grid_proj_storageT<<<BLOCK_NUM,THREADS_NUM>>>(tsdf_gpu_storage, R_GPU, K_GPU, depth_gpu, XYZimage_gpu, 
                                                      transform_grid_gpu, grid_size, truncated_value, width, height);
    
    // delete[] p;
    // p = new float[grid_size*3];
    // checkCUDA(__LINE__, cudaMemcpy(p, tsdf_gpu, sizeof(float)*grid_size*3, cudaMemcpyDeviceToHost));
    // fwrite(p, sizeof(float), grid_size*3, fp);
    // fclose(fp);
    // delete[] p;
  }

  ~DepthImage()
  {
    cout << "Calling destructive function" << endl;

    if (depth_cpu) delete[] depth_cpu;
    // if (K_CPU) delete[] K_CPU;
    // if (R_CPU) delete[] R_CPU;
    
    if (K_GPU!=NULL) checkCUDA(__LINE__, cudaFree(K_GPU));
    if (R_GPU!=NULL) checkCUDA(__LINE__, cudaFree(R_GPU));
    if (tsdf_gpu!=NULL) checkCUDA(__LINE__, cudaFree(tsdf_gpu));
    if (tsdf_gpu_storage!=NULL) checkCUDA(__LINE__, cudaFree(tsdf_gpu_storage));
    if (scene_setting_gpu!=NULL) checkCUDA(__LINE__, cudaFree(scene_setting_gpu));
    if (depth_gpu!=NULL) checkCUDA(__LINE__, cudaFree(depth_gpu));
    if (XYZimage_gpu!=NULL) checkCUDA(__LINE__, cudaFree(XYZimage_gpu));

    if (regular_grid_gpu!=NULL) checkCUDA(__LINE__, cudaFree(regular_grid_gpu));
    if (transform_grid_gpu!=NULL) checkCUDA(__LINE__, cudaFree(transform_grid_gpu));
  }
};



class SceneTemplate{
public:
  vector<Wall3D> walls;
  vector<Box3D> objects;
  vector<unsigned int> wall_valid;
  vector<unsigned int> object_valid;
  SceneTemplate()
  {
    walls.clear();
    objects.clear();
    wall_valid.clear();
    object_valid.clear();
  }



  void transformTemplate(float* Rot, float* Tsl)
  {
    for (int i=0; i<walls.size(); i++)
    {
      walls[i].transformWall3D(Rot, Tsl);
    }
    for (int i=0; i<objects.size(); i++)
    {
      objects[i].transformBox3D(Rot, Tsl);
    }
  }

  void disturbTemplate( SceneTemplate &avg_temp, SceneTemplate &std_temp, std::mt19937 &rng_)
  {
    // std::default_random_engine generator(rng_);
    std::normal_distribution<float> distribution(0.0,1.0);

    // for (int i = 0; i < 10; ++i)
    // {
    //   float val = distribution( rng_ );
    //   cout << val * std_temp.objects[0].center[0] << " ";  
    // }
    // cout << endl;

    // int count = 0;
    for (int i = 0; i<walls.size(); i++)
    {
      for (int j=0; j<2; j++)
      {
        if ( walls[i].type != 0 && j != 0 ) continue; // fix adding noise to wall
        float val = distribution( rng_ );
        walls[i].top_loc[j] = val * std_temp.walls[i].top_loc[j]*0.2 + avg_temp.walls[i].top_loc[j];

        // disturb_noise[count] = walls[i].points[j];
        // count ++;
      }
    }

    for (int i = 0; i<objects.size(); i++)
    {
      // if ( i == 0 )
      // {
      //   cout 
      // }

      for (int j=0; j<3; j++)
      {
        float val = distribution( rng_ );
        objects[i].center[j] = val * std_temp.objects[i].center[j]*0.2 + avg_temp.objects[i].center[j];
        // disturb_noise[count] = objects[i].center[j];
        // count ++;
      }
      for(int j=0; j<3; j++)
      {
        float val = distribution( rng_ );
        objects[i].coeff[j] = val * std_temp.objects[i].coeff[j]*0.2 + avg_temp.objects[i].coeff[j];
        // disturb_noise[count] = objects[i].coeff[j];
        // count ++;
      }
    }
  }

  int numel_diff()
  {
    int dim = 0;
    // cout << "walls: " << walls.size() << endl;
    // cout << "objects: " << objects.size() << endl;
    for (int i=0; i<walls.size(); i++)
    {
      if (walls[i].type == 1 || walls[i].type == 2)
      {
        dim += 1;
      }
      else if (walls[i].type == 0)
      {
        dim += 2;
      }
    }
    for (int i=0; i<objects.size(); i++)
    {
      // dim += 7;
      dim += 6;
    }
    return dim;
  }

  int numel_cls()
  {
    return walls.size() + objects.size();
  }

  void getRegressionValue( SceneTemplate &t, float* diff)
  {
    // diff = new float[numel_diff()];
    int count = 0;
    for (int i=0; i<walls.size(); i++)
    {
      if (walls[i].type == 1 || walls[i].type == 2)
      {
        diff[count] = walls[i].top_loc[0] - t.walls[i].top_loc[0];
        count ++;
      }
      else if (walls[i].type == 0)
      {
        diff[count] = walls[i].top_loc[0] - t.walls[i].top_loc[0];
        count++;
        diff[count] = walls[i].top_loc[1] - t.walls[i].top_loc[1];
        count++;
      }
    }
    for (int i=0; i<objects.size(); i++)
    {
      diff[count] = objects[i].center[0] - t.objects[i].center[0];
      count++;
      diff[count] = objects[i].center[1] - t.objects[i].center[1];
      count++;
      diff[count] = objects[i].center[2] - t.objects[i].center[2];
      count++;
      diff[count] = objects[i].coeff[0] - t.objects[i].coeff[0];
      count++;
      diff[count] = objects[i].coeff[1] - t.objects[i].coeff[1];
      count++;
      diff[count] = objects[i].coeff[2] - t.objects[i].coeff[2];
      count++;
      // diff[count] = objects[i].orientation[0] * t.objects[i].orientation[1] - objects[i].orientation[1] * t.objects[i].orientation[0];
      // count++;
    }

    // cout << "Regression Value: " << endl;
    // for (int i = 0; i < count; ++i)
    // {
    //   cout << diff[i] << " ";
    // }
    // cout << endl;
  }

  void getRegressionValue_storage( SceneTemplate &t, StorageT* diff)
  {
    // diff = new float[numel_diff()];
    int count = 0;
    for (int i=0; i<walls.size(); i++)
    {
      if (walls[i].type == 1 || walls[i].type == 2)
      {
        diff[count] = CPUCompute2StorageT(ComputeT(walls[i].top_loc[0] - t.walls[i].top_loc[0]));
        count ++;
      }
      else if (walls[i].type == 0)
      {
        diff[count] = CPUCompute2StorageT(ComputeT(walls[i].top_loc[0] - t.walls[i].top_loc[0]));
        count++;
        diff[count] = CPUCompute2StorageT(ComputeT(walls[i].top_loc[1] - t.walls[i].top_loc[1]));
        count++;
      }
    }
    for (int i=0; i<objects.size(); i++)
    {
      diff[count] = CPUCompute2StorageT(ComputeT(objects[i].center[0] - t.objects[i].center[0]));
      count++;
      diff[count] = CPUCompute2StorageT(ComputeT(objects[i].center[1] - t.objects[i].center[1]));
      count++;
      diff[count] = CPUCompute2StorageT(ComputeT(objects[i].center[2] - t.objects[i].center[2]));
      count++;
      diff[count] = CPUCompute2StorageT(ComputeT(objects[i].coeff[0] - t.objects[i].coeff[0]));
      count++;
      diff[count] = CPUCompute2StorageT(ComputeT(objects[i].coeff[1] - t.objects[i].coeff[1]));
      count++;
      diff[count] = CPUCompute2StorageT(ComputeT(objects[i].coeff[2] - t.objects[i].coeff[2]));
      count++;
      // diff[count] = CPUCompute2StorageT(ComputeT(objects[i].orientation[0] * t.objects[i].orientation[1] - objects[i].orientation[1] * t.objects[i].orientation[0]));
      // count++;
    }

    // cout << "Regression Value: " << endl;
    // for (int i = 0; i < count; ++i)
    // {
    //   cout << diff[i] << " ";
    // }
    // cout << endl;
  }

  void getRegressionBinary(float* diff)
  {
    // diff = new float[numel_diff()];
    int count = 0;
    for (int i=0; i<walls.size(); i++)
    {
      if (walls[i].type == 1 || walls[i].type == 2)
      {
        diff[count] = wall_valid[i];
        count ++;
      }
      else if (walls[i].type == 0)
      {
        diff[count] = wall_valid[i];
        count++;
        diff[count] = wall_valid[i];
        count++;
      }
    }
    for (int i=0; i<objects.size(); i++)
    {
      diff[count] = object_valid[i];
      count++;
      diff[count] = object_valid[i];
      count++;
      diff[count] = object_valid[i];
      count++;
      diff[count] = object_valid[i];
      count++;
      diff[count] = object_valid[i];
      count++;
      diff[count] = object_valid[i];
      count++;
      // diff[count] = object_valid[i];
      // count++;
    }

    // cout << "Regression Binary: " << endl;
    // for (int i = 0; i < count; ++i)
    // {
    //   cout << diff[i] << " ";
    // }
    // cout << endl;
  }

  void getRegressionBinary_storage(StorageT* diff)
  {
    // diff = new float[numel_diff()];
    int count = 0;
    for (int i=0; i<walls.size(); i++)
    {
      if (walls[i].type == 1 || walls[i].type == 2)
      {
        diff[count] = CPUCompute2StorageT(ComputeT(wall_valid[i]));
        count ++;
      }
      else if (walls[i].type == 0)
      {
        diff[count] = CPUCompute2StorageT(ComputeT(wall_valid[i]));
        count++;
        diff[count] = CPUCompute2StorageT(ComputeT(wall_valid[i]));
        count++;
      }
    }
    for (int i=0; i<objects.size(); i++)
    {
      diff[count] = CPUCompute2StorageT(ComputeT(object_valid[i]));
      count++;
      diff[count] = CPUCompute2StorageT(ComputeT(object_valid[i]));
      count++;
      diff[count] = CPUCompute2StorageT(ComputeT(object_valid[i]));
      count++;
      diff[count] = CPUCompute2StorageT(ComputeT(object_valid[i]));
      count++;
      diff[count] = CPUCompute2StorageT(ComputeT(object_valid[i]));
      count++;
      diff[count] = CPUCompute2StorageT(ComputeT(object_valid[i]));
      count++;
      // diff[count] = CPUCompute2StorageT(ComputeT(object_valid[i]));
      // count++;
    }

    // cout << "Regression Binary: " << endl;
    // for (int i = 0; i < count; ++i)
    // {
    //   cout << diff[i] << " ";
    // }
    // cout << endl;
  }

  void getClassificationValue(float* diff)
  {
    int count = 0;
    for (int i=0; i<walls.size(); i++)
    {
      diff[count] = wall_valid[i];
      count ++;
    }
    for (int i=0; i<objects.size(); i++)
    {
      diff[count] = object_valid[i];
      count++;
    } 

    // cout << "Classification Value: " << endl;
    // for (int i = 0; i < count; ++i)
    // {
    //   cout << diff[i] << " ";
    // }
    // cout << endl;
  }

  void getClassificationValue_storage(StorageT* diff)
  {
    int count = 0;
    for (int i=0; i<walls.size(); i++)
    {
      diff[count] = CPUCompute2StorageT(ComputeT(wall_valid[i]));
      count ++;
    }
    for (int i=0; i<objects.size(); i++)
    {
      diff[count] = CPUCompute2StorageT(ComputeT(object_valid[i]));
      count++;
    } 

    // cout << "Classification Value: " << endl;
    // for (int i = 0; i < count; ++i)
    // {
    //   cout << diff[i] << " ";
    // }
    // cout << endl;
  }

  int regress_len( int uni_id)
  {
    if (uni_id < walls.size())
    {
      if (walls[uni_id].type == 0)
      {
        return 2;
      }
      else
      {
        return 1;
      }
    }
    else
    {
      return 6;
    }
  }

  void getIOUbased_value_storage( vector<StorageT*> cls_val, vector<StorageT*> reg_bin, vector<StorageT*> reg_val, vector<StorageT*> &boxes, SceneTemplate &gnd, SceneTemplate &norm, vector<float> spatial_range, vector<float> grid_size)
  {
    int wall_num = walls.size();
    int object_num = objects.size();
    vector<float> resolution;
    resolution.clear();
    resolution.push_back( (spatial_range[1]-spatial_range[0])/grid_size[0] );
    resolution.push_back( (spatial_range[3]-spatial_range[2])/grid_size[1] );
    resolution.push_back( (spatial_range[5]-spatial_range[4])/grid_size[2] );
    // cout << "Resolution: " << vecPrintString(resolution) << endl;

    for (int i = 0; i<wall_num; i++)
    {
      StorageT* t_cls_val = cls_val[i];
      StorageT* t_reg_bin = reg_bin[i];
      StorageT* t_reg_val = reg_val[i];
      StorageT* t_boxes   = boxes[i];
      // cout << "load: " << gnd.wall_valid[i] << endl;
      // if (i >= gnd.wall_valid.size())
      // {
      //   cout << "Warning: wall memory exceed! Shouldn't appear during training!" << endl;
      // }

      if (walls[i].type==0)
      {        
        t_cls_val[0] = CPUCompute2StorageT(ComputeT(gnd.wall_valid[i]));
        t_reg_bin[0] = CPUCompute2StorageT(ComputeT(gnd.wall_valid[i]));
        t_reg_bin[1] = CPUCompute2StorageT(ComputeT(gnd.wall_valid[i]));
        t_reg_val[0] = CPUCompute2StorageT(ComputeT(walls[i].top_loc[0] - gnd.walls[i].top_loc[0]));
        t_reg_val[1] = CPUCompute2StorageT(ComputeT(walls[i].top_loc[1] - gnd.walls[i].top_loc[1]));
        if( abs(walls[i].top_loc[0]) < abs(walls[i].top_loc[1]) ) // norm Y
        {
          t_boxes[0] = CPUCompute2StorageT(ComputeT(0));
          // t_boxes[1] = CPUCompute2StorageT(ComputeT(31));
          t_boxes[1] = CPUCompute2StorageT(ComputeT(grid_size[0]));
          t_boxes[2] = CPUCompute2StorageT(ComputeT((walls[i].top_loc[1] - spatial_range[2])/resolution[1] - 5));
          t_boxes[3] = CPUCompute2StorageT(ComputeT((walls[i].top_loc[1] - spatial_range[2])/resolution[1] + 5));
          t_boxes[4] = CPUCompute2StorageT(ComputeT(0));
          // t_boxes[5] = CPUCompute2StorageT(ComputeT(63));
          t_boxes[5] = CPUCompute2StorageT(ComputeT(grid_size[2]));
        }
        else // norm X
        {
          t_boxes[0] = CPUCompute2StorageT(ComputeT(0));
          // t_boxes[1] = CPUCompute2StorageT(ComputeT(31));
          t_boxes[1] = CPUCompute2StorageT(ComputeT(grid_size[0]));
          t_boxes[2] = CPUCompute2StorageT(ComputeT(0));
          // t_boxes[3] = CPUCompute2StorageT(ComputeT(63));
          t_boxes[3] = CPUCompute2StorageT(ComputeT(grid_size[1]));
          t_boxes[4] = CPUCompute2StorageT(ComputeT((walls[i].top_loc[0] - spatial_range[4])/resolution[2] - 5));
          t_boxes[5] = CPUCompute2StorageT(ComputeT((walls[i].top_loc[0] - spatial_range[4])/resolution[2] + 5));         
        }
      }
      else
      {
        t_cls_val[0] = CPUCompute2StorageT(ComputeT(gnd.wall_valid[i]));
        t_reg_bin[0] = CPUCompute2StorageT(ComputeT(gnd.wall_valid[i]));
        t_reg_val[0] = CPUCompute2StorageT(ComputeT(walls[i].top_loc[0] - gnd.walls[i].top_loc[0]));
        t_boxes[0] = CPUCompute2StorageT(ComputeT((walls[i].top_loc[0] - spatial_range[0])/resolution[0] - 5));
        t_boxes[1] = CPUCompute2StorageT(ComputeT((walls[i].top_loc[0] - spatial_range[0])/resolution[0] + 5));
        t_boxes[2] = CPUCompute2StorageT(ComputeT(0));
        // t_boxes[3] = CPUCompute2StorageT(ComputeT(63));
        t_boxes[3] = CPUCompute2StorageT(ComputeT(grid_size[1]));
        t_boxes[4] = CPUCompute2StorageT(ComputeT(0));
        // t_boxes[5] = CPUCompute2StorageT(ComputeT(63));
        t_boxes[5] = CPUCompute2StorageT(ComputeT(grid_size[2]));
      }
    }

    for (int i = 0; i<object_num; i++)
    {
      int oi = i + walls.size();
      StorageT* t_cls_val = cls_val[oi];
      StorageT* t_reg_bin = reg_bin[oi];
      StorageT* t_reg_val = reg_val[oi];
      StorageT* t_boxes   = boxes[oi];

      // if (i >= gnd.object_valid.size())
      // {
      //   cout << "Warning: object memory exceed! Shouldn't appear during training!" << endl;
      //   cout << "Correct size: " << gnd.objects.size() << endl;
      //   for (int j = 0; j<gnd.object_valid.size(); j++)
      //   {
      //     cout << gnd.object_valid[j] << " ";
      //   }
      //   for (int j = gnd.object_valid.size(); j<=i; j++)
      //   {
      //     cout << "e" << gnd.object_valid[j] << " ";
      //   }
      //   cout << endl;
      // }

      // std::cout << i << std::endl;
      float iou = objects[i].IOU_MW(gnd.objects[i]);
      if ( iou > 0.1 && gnd.object_valid[i] )  t_cls_val[0] = CPUCompute2StorageT(ComputeT(1));
      else t_cls_val[0] = CPUCompute2StorageT(ComputeT(0));
      // t_cls_val[0] = CPUCompute2StorageT(ComputeT(iou));
      for (int j = 0; j < 6; ++j)
      {
        // t_reg_bin[j] = t_cls_val[0];
        if (gnd.object_valid[i])
          t_reg_bin[j] = CPUCompute2StorageT(ComputeT(1));
        else
          t_reg_bin[j] = CPUCompute2StorageT(ComputeT(0));
      }

      t_reg_val[0] = CPUCompute2StorageT(ComputeT((objects[i].center[0] - gnd.objects[i].center[0])/(norm.objects[i].center[0]+0.01)));
      t_reg_val[1] = CPUCompute2StorageT(ComputeT((objects[i].center[1] - gnd.objects[i].center[1])/(norm.objects[i].center[1]+0.01)));
      t_reg_val[2] = CPUCompute2StorageT(ComputeT((objects[i].center[2] - gnd.objects[i].center[2])/(norm.objects[i].center[2]+0.01)));
      t_reg_val[3] = CPUCompute2StorageT(ComputeT((objects[i].coeff[0]  - gnd.objects[i].coeff[0])/(norm.objects[i].coeff[0]+0.01)));
      t_reg_val[4] = CPUCompute2StorageT(ComputeT((objects[i].coeff[1]  - gnd.objects[i].coeff[1])/(norm.objects[i].coeff[1]+0.01)));
      t_reg_val[5] = CPUCompute2StorageT(ComputeT((objects[i].coeff[2]  - gnd.objects[i].coeff[2])/(norm.objects[i].coeff[2]+0.01)));
      
      float c_boxes[6];
      objects[i].floorRectangle_MW(c_boxes);
      t_boxes[0] = CPUCompute2StorageT(ComputeT( (c_boxes[0]-spatial_range[0])/resolution[0] ));
      t_boxes[1] = CPUCompute2StorageT(ComputeT( (c_boxes[1]-spatial_range[0])/resolution[0] ));
      t_boxes[2] = CPUCompute2StorageT(ComputeT( (c_boxes[2]-spatial_range[2])/resolution[1] ));
      t_boxes[3] = CPUCompute2StorageT(ComputeT( (c_boxes[3]-spatial_range[2])/resolution[1] ));
      t_boxes[4] = CPUCompute2StorageT(ComputeT( (c_boxes[4]-spatial_range[4])/resolution[2] ));
      t_boxes[5] = CPUCompute2StorageT(ComputeT( (c_boxes[5]-spatial_range[4])/resolution[2] ));
      // for (int j = 0; j < 6; ++j)
      // {
      //   t_boxes[j] = CPUCompute2StorageT(ComputeT(c_boxes[j]));
      // }
    }


    // int count = 0;
    // for (int i=0; i<walls.size(); i++)
    // {
    //   cls_val[count] = CPUCompute2StorageT(ComputeT(wall_valid[i]));
    //   count ++;
    // }
    // for (int i=0; i<objects.size(); i++)
    // {
    //   float iou = objects[i].IOU(gnd.objects[i]);
    //   if ( iou < 0.3 )  cls_val[count] = 1;
    //   else  cls_val[count] = 0;
    //   count ++;
    // }
    // int count = 0;
    // for (int i=0; i<walls.size(); i++)
    // {
    //   if (walls[i].type == 1 || walls[i].type == 2)
    //   {
    //     reg_bin[count] = CPUCompute2StorageT(ComputeT(wall_valid[i]));
    //     count ++;
    //   }
    //   else if (walls[i].type == 0)
    //   {
    //     reg_bin[count] = CPUCompute2StorageT(ComputeT(wall_valid[i]));
    //     count++;
    //     reg_bin[count] = CPUCompute2StorageT(ComputeT(wall_valid[i]));
    //     count++;
    //   }
    // }
    // for (int i=0; i<objects.size(); i++)
    // {
    //   reg_bin[count] = CPUCompute2StorageT(ComputeT(cls_val[ i + walls.size() ]));
    //   count++;
    //   reg_bin[count] = CPUCompute2StorageT(ComputeT(cls_val[ i + walls.size() ]));
    //   count++;
    //   reg_bin[count] = CPUCompute2StorageT(ComputeT(cls_val[ i + walls.size() ]));
    //   count++;
    //   reg_bin[count] = CPUCompute2StorageT(ComputeT(cls_val[ i + walls.size() ]));
    //   count++;
    //   reg_bin[count] = CPUCompute2StorageT(ComputeT(cls_val[ i + walls.size() ]));
    //   count++;
    //   reg_bin[count] = CPUCompute2StorageT(ComputeT(cls_val[ i + walls.size() ]));
    //   count++;
    // }
    // getRegressionValue_storage( gnd, reg_val);


  }

  void loadTemplate(FILE* &fp)
  {
    walls.clear();
    objects.clear();
    wall_valid.clear();
    object_valid.clear();

    unsigned int num_object_all;
    fread(&num_object_all, 1, sizeof(unsigned int), fp);
    for (int i = 0; i<num_object_all; ++i)
    {
      unsigned int is_object;
      fread(&is_object, 1, sizeof(unsigned int), fp);
      if (is_object==0)
      {
        walls.push_back(Wall3D());
        walls[walls.size()-1].load(fp);
      }
      else if (is_object==1)
      {
        objects.push_back(Box3D());
        objects[objects.size()-1].load(fp);
      }
    }

    unsigned int *p = new unsigned int[walls.size()+objects.size()];
    fread((void*)p, walls.size()+objects.size(), sizeof(unsigned int), fp);
    int count = 0;
    // cout << "wall_valid: ";
    for (int i=0; i<walls.size(); i++,count++)
    {
      wall_valid.push_back(p[count]);
    }
    // for (int i=0; i<walls.size(); i++)
    // {
    //   cout << " " << wall_valid[i];
    // }
    // cout << endl;
    // cout << "object_valid: ";
    for (int i=0; i<objects.size(); i++,count++)
    {
      object_valid.push_back(p[count]);
    }
    // for (int i=0; i<objects.size(); i++)
    // {
    //   cout << " " << object_valid[i];
    // }
    // cout << endl;
  }

};


class Scene3D{
public:
  // defined in .list file
  string filename;
  float K[9];
  float R[9];
  float Rot[4];
  float Tsl[3];
  unsigned int width;
  unsigned int height;
  SceneTemplate gnd;
  
  void alignTemplate( SceneTemplate &t )
  {


  }

  void loadDepthImage( DepthImage &depth)
  {
    depth.readImage(filename);
  }

  void loadCameraInfo(FILE* &fp)
  {
    fread((void*)Rot, sizeof(float), 4, fp);
    fread((void*)Tsl, sizeof(float), 3, fp);
    fread((void*)R,   sizeof(float), 9, fp);
    fread((void*)K,   sizeof(float), 9, fp);
    fread((void*)(&height), sizeof(unsigned int), 1, fp);
    fread((void*)(&width),  sizeof(unsigned int), 1, fp);
    
    // cout << "K: ";
    // for (int i = 0; i<3; i++)
    // {
    //   cout << Tsl[i] << " ";
    // }
    // cout << endl;
    // cout << height << " " << width << endl;
  }

  void loadTemplate(FILE* &fp)
  {
    gnd.loadTemplate(fp);
  }

  // void loadMetaData(FILE* &fp)
  // {
  //   unsigned int len = 0;
  //   // cout << "len: " << len << endl;
  //   // fread( (void*)(&len), sizeof(unsigned int), 1, fp); 
  //   // fread( (void*)(filename.data()), sizeof(char), len, fp);
  //   // cout << "len: " << len << ", filename: " << filename << endl;

  //   fread( (void*)K, sizeof(float), 9, fp);
  //   for (int i = 0; i<9; i++)
  //   {
  //     cout << K[i] << " ";
  //   }
  //   fread( (void*)R, sizeof(float), 9, fp);
  //   for (int i = 0; i<9; i++)
  //   {
  //     cout << R[i] << " ";
  //   }

  //   fread( (void*)(&len), sizeof(unsigned int), 1, fp); 
  //   cout << "len: " << len << endl;
  //   for (int i=0; i<len; i++)
  //   {
  //     unsigned int entity_type;
  //     fread( (void*)(&entity_type), sizeof(unsigned int), 1, fp);
  //     cout << "entity_type: " <<  entity_type << endl;
  //     if ( entity_type==0 )
  //     {
  //       walls.push_back(Wall3D());
  //       walls[walls.size()-1].load(fp);
  //     }
  //     else if ( entity_type==1 )
  //     {
  //       objects.push_back(Box3D());
  //       objects[objects.size()-1].load(fp);
  //     }
  //   }
  //   fread((void*)(&height), sizeof(unsigned int), 1, fp);
  //   fread((void*)(&width), sizeof(unsigned int), 1, fp);
  // }

  Scene3D() {
    width = 0;
    height = 0;
  }

  // Scene3D(FILE* &fp) {
  //   loadMetaData(fp);
  // }  
  
};

class DataManager{
public:
  void init( vector<int> &data_ids, vector<int> &labels, int num_label_, std::mt19937& rng_, bool balance_, bool shuffle_ )
  {
    if (balance_)
    {
      num_label = num_label_;

      cls_counter.assign(num_label, 0);
      iterator = 0;

      cls_ids.assign(num_label, vector<int>(0,0));
      for (int i = 0; i < data_ids.size(); ++i)
      {
        int id = data_ids[i];
        cls_ids[labels[id]].push_back(i);
      }

      cls_order.assign(num_label, vector<size_t>(0,0));    
    }
    else
    {
      num_label = 1;
      
      cls_counter.assign(num_label, 0);
      iterator = 0;

      cls_ids.assign(num_label, vector<int>(0,0));
      cls_ids[0].resize( data_ids.size());
      for (int i = 0; i < data_ids.size(); ++i)
      {
        cls_ids[0][i] = i;
      }

      cls_order.assign(num_label, vector<size_t>(0,0));
    }

    rng = rng_;
    shuffle = shuffle_;
    shuffle_all();
  }

  void shuffle_all( )
  {
    for (int i = 0; i < num_label; ++i)
    {
      shuffle_cls(i);
    }
  }

  void shuffle_cls( int cls )
  {
    if (shuffle)
      cls_order[cls] = randperm( cls_ids[cls].size(), rng);
    else {
      cls_order[cls].resize( cls_ids[cls].size() );
      for (int i = 0; i < cls_ids[cls].size(); ++i ) cls_order[cls][i] = i;
    }
  }

  int next_id( int &epoch)
  {
    // cout << "iterator: " << iterator << "; cls_counter: " << cls_counter[iterator] << "; cls_order: " << cls_order[iterator][cls_counter[iterator]] <<endl;

    int id1 = cls_order[iterator][cls_counter[iterator]];
    int next_id = cls_ids[iterator][id1];

    cls_counter[iterator] ++;
    if ( cls_counter[iterator] >= cls_order[iterator].size() )
    {
      cls_counter[iterator] = 0;
      shuffle_cls(iterator);
      if (num_label==1) epoch++;
    }

    iterator ++;
    if ( iterator >= num_label )
    {
      iterator = 0;
    }

    return next_id;
  }

  vector<vector<int>> cls_ids;
  vector<vector<size_t>> cls_order;
  vector<int> cls_counter;
  int num_label;
  int iterator;
  bool shuffle;
  std::mt19937 rng;
};


///////////// new layer
// Yinda: a layer read depth image, and compute tsdf online
class SceneHolisticOnlineTSDFROILayer : public DataLayer {
public:
  int epoch_prefetch;

  string data_root;
  vector<string> data_list_name;
  vector<int> data_list_id;
  vector<int> label_list;
  string template_avg_file;
  string template_std_file;
  string file_list;
  string file_label;
  string ground_truth_file;
  string camera_info_file;  
  vector<int> grid_size;
  vector<float> spatial_range;
  vector<float> noise_scale;
  
  int batch_size;
  future<void> lock;

  FILE* dataFILE;
  vector<size_t> ordering;

  int template_id;
  vector<SceneTemplate> template_avg;
  vector<SceneTemplate> template_std;

  DepthImage depth_image;
  vector<Scene3D> ground_truth;

  unsigned int num_scene_template;
  unsigned int num_scene_3d;
  
  vector<int> data_dims;
  vector<vector<int>> object_reg_sz;
  vector<vector<int>> object_cls_sz;
  vector<vector<int>> object_box_sz;
  vector<int> noise_value_dims;

  StorageT* data_gpu;
  StorageT* noise_value_gpu;
  StorageT* noise_value_cpu;

  bool disturb = false;

  int numel_batch_data;
  int numel_single_data;
  // int numel_batch_bbox_regression;
  // int numel_single_bbox_regression;
  // int numel_batch_bbox_classification;
  // int numel_single_bbox_classification;
  int numel_batch_noise;
  int numel_single_noise;
  // int numel_batch_scene_classification;
  // int numel_single_scene_classification;
  // int numel_batch_bbox_c_classification;
  // int numel_single_bbox_c_classification;

  vector<StorageT*> object_cls_val_cpu;
  vector<StorageT*> object_reg_bin_cpu;
  vector<StorageT*> object_reg_val_cpu;
  vector<StorageT*> object_box_cpu;

  vector<StorageT*> object_cls_val_gpu;
  vector<StorageT*> object_reg_bin_gpu;
  vector<StorageT*> object_reg_val_gpu;
  vector<StorageT*> object_box_gpu;

  vector<float> grid_size_reverse;
  vector<float> spatial_reverse;

  int object_num;

  bool shuffle_data;
  bool balance_data;

  bool use_predef;
  string predef_noise_file;
  string predef_anchor_file;
  float *predef_noise;
  vector<SceneTemplate> predef_anchor;

  DataManager id_manager;

  int numofitems(){
    return data_list_name.size();
  };

  int numofitemsTruncated(){
    return batch_size * floor(double(numofitems())/double(batch_size));
  };

  void init(){
    cout << "Data layer init..." << endl;
    train_me = false;

    //////////////////// read training data id
    epoch_prefetch  = 0;
    dataFILE = fopen( file_list.c_str(), "r");
    if (dataFILE==NULL){
      cerr<<"Fail to open the data file"<<endl;
      FatalError(__LINE__);
    }

    data_list_name.clear();
    data_list_id.clear();
    int iBuffer;
    char cBuffer[10];
    int num_data;
    iBuffer = fscanf( dataFILE, "%d", &num_data);
    if (iBuffer != 1)
    {
      cerr<<"Fail to read properly"<<endl;
      FatalError(__LINE__);
    }
    for ( int i = 0; i<num_data; ++i )
    {
      int readcount = fscanf( dataFILE, "%s", cBuffer);
      string s(cBuffer);
      data_list_name.push_back(s);
      data_list_id.push_back(atoi(cBuffer)-1);
    }
    fclose(dataFILE);
    cout << data_list_name.size() << "data load!" <<  endl;
    shuffle();
    counter = 0;
    //////////////////// read training data id end

    ////////////////////// read label list
    dataFILE = fopen( file_label.c_str(), "r");
    label_list.clear();
    fscanf( dataFILE, "%d", &num_data);
    for ( int i=0; i<num_data; ++i )
    {
      fscanf( dataFILE, "%d", &iBuffer);
      label_list.push_back(iBuffer);
    }
    fclose(dataFILE);
    cout << label_list.size() << " label loaded!" << endl;
    //////////////////// read label list end

    ////////////////////// read scene template
    cout << "Read template..." << endl;
    dataFILE = fopen(template_avg_file.c_str(), "rb");
    if (dataFILE==NULL){
      cerr<<"Fail to open the data file"<<endl;
      FatalError(__LINE__);
    }
    fread( (void*)(&num_scene_template), sizeof(unsigned int), 1, dataFILE);
    cout << "num_scene_template: " << num_scene_template << endl;
    template_avg.assign(num_scene_template, SceneTemplate());
    for (int sid=0; sid<num_scene_template; sid++)
    {
      template_avg[sid].loadTemplate(dataFILE);
    }
    fclose(dataFILE);

    dataFILE = fopen(template_std_file.c_str(), "rb");
    if (dataFILE==NULL){
      cerr<<"Fail to open the data file"<<endl;
      FatalError(__LINE__);
    }
    fread( (void*)(&num_scene_template), sizeof(unsigned int), 1, dataFILE);
    cout << "num_scene_template: " << num_scene_template << endl;
    template_std.assign(num_scene_template, SceneTemplate());
    for (int sid=0; sid<num_scene_template; sid++)
    {
      template_std[sid].loadTemplate(dataFILE);
    }
    fclose(dataFILE);
    //////////////////// read scene template end

    ////////////////////// read ground truth
    cout << "Read ground truth file ..." << endl;
    dataFILE = fopen( ground_truth_file.c_str(), "rb");
    fread( (void*)(&num_scene_3d), sizeof(unsigned int), 1, dataFILE);
    cout << "num_scene_3d: " << num_scene_3d << endl;
    ground_truth.assign(num_scene_3d, Scene3D());
    // ground_truth.clear();
    cout << "Initialize..." << endl;
    for (int sid=0; sid<num_scene_3d; sid++)
    // for (int sid=0; sid<1; sid++)
    {
      ground_truth[sid].loadTemplate(dataFILE);
    }
    fclose(dataFILE);
    ///////////////////// read ground truth end

    ////////////////////// read camera parameters
    cout << "Read camera parameter file ..." << endl;
    dataFILE = fopen( camera_info_file.c_str(), "rb");
    int checkNum;
    fread( (void*)(&checkNum), sizeof(unsigned int), 1, dataFILE);
    if (checkNum != num_scene_3d )
    {
      cerr<<"Data number inconsistent"<<endl;
      FatalError(__LINE__);
    }

    cout << "Number camera: " << checkNum << endl;
    for (int sid=0; sid<num_scene_3d; sid++)
    {
      cout << sid << endl;
      ground_truth[sid].loadCameraInfo(dataFILE);
    }
    fclose(dataFILE);
    ////////////////////// read camera parameter end

    ////////////////////// Prepare for TSDF computing
    float temp[10];
    temp[0] = spatial_range[0]; temp[1] = spatial_range[1];
    temp[2] = spatial_range[2]; temp[3] = spatial_range[3];
    temp[4] = spatial_range[4]; temp[5] = spatial_range[5];
    temp[6] = grid_size[3]; temp[7] = grid_size[2]; temp[8] = grid_size[1];
    temp[9] = 3.0 * (spatial_range[1] - spatial_range[0]) / grid_size[3];
    cout << "temp[9] = " << temp[9] << endl;
    depth_image.SceneSetting(temp);

    grid_size_reverse.clear();
    grid_size_reverse.push_back( grid_size[1]);
    grid_size_reverse.push_back( grid_size[2]);
    grid_size_reverse.push_back( grid_size[3]);
    spatial_reverse.clear();
    spatial_reverse.push_back( spatial_range[4] );
    spatial_reverse.push_back( spatial_range[5] );
    spatial_reverse.push_back( spatial_range[2] );
    spatial_reverse.push_back( spatial_range[3] );
    spatial_reverse.push_back( spatial_range[0] );
    spatial_reverse.push_back( spatial_range[1] );

    // cout << "grid_size_reverse: " << vecPrintString(grid_size_reverse) << endl;
    // cout << "spatial_reverse: " << vecPrintString(spatial_reverse) << endl;

    ////////////////////// Set essential memory size
    data_dims.clear();
    data_dims.push_back(batch_size);
    data_dims.push_back(grid_size[0]);
    data_dims.push_back(grid_size[1]);
    data_dims.push_back(grid_size[2]);
    data_dims.push_back(grid_size[3]);
    numel_batch_data = numel(data_dims);
    numel_single_data = sizeofitem(data_dims);
    cout << "data_dims: " << vecPrintString(data_dims) << endl;
    cout << "numel_batch_data: " << numel_batch_data << endl;
    cout << "numel_single_data: " << numel_single_data << endl;

    SceneTemplate chosen_template_avg = template_avg[template_id];
    SceneTemplate chosen_template_std = template_std[template_id];
    object_num = chosen_template_std.walls.size() + chosen_template_std.objects.size();
    cout << "Object number: " << object_num << endl;

    object_reg_sz.assign(object_num, vector<int>(5,1));
    object_cls_sz.assign(object_num, vector<int>(5,1));  
    object_box_sz.assign(object_num, vector<int>(5,1));  
    for (int i = 0; i < object_num; ++i)
    {
      object_reg_sz[i][0] = batch_size;
      object_reg_sz[i][1] = chosen_template_std.regress_len(i);
      cout << "regression len: " << object_reg_sz[i][1] << endl;
      object_cls_sz[i][0] = batch_size;
      object_box_sz[i][0] = batch_size;
      object_box_sz[i][1] = 7;
    }

    noise_value_dims.clear();
    noise_value_dims.push_back(batch_size);
    noise_value_dims.push_back(3);
    noise_value_dims.push_back(1);
    noise_value_dims.push_back(1);
    noise_value_dims.push_back(1);
    numel_batch_noise = numel(noise_value_dims);
    numel_single_noise = sizeofitem(noise_value_dims);
    cout << "noise_value_dims: " << vecPrintString(noise_value_dims) << endl;
    cout << "numel_batch_noise: " << numel_batch_noise << endl;
    cout << "numel_single_noise: " << numel_single_noise << endl;

    /////////////////// Allocate memory
    object_cls_val_cpu.assign(object_num, NULL);
    object_reg_bin_cpu.assign(object_num, NULL);
    object_reg_val_cpu.assign(object_num, NULL);
    object_box_cpu.assign(object_num, NULL);

    for (int i = 0; i < object_cls_sz.size(); ++i)
    {
      object_cls_val_cpu[i] = new StorageT[numel(object_cls_sz[i])];
      object_reg_bin_cpu[i] = new StorageT[numel(object_reg_sz[i])];
      object_reg_val_cpu[i] = new StorageT[numel(object_reg_sz[i])];
      object_box_cpu[i] = new StorageT[numel(object_box_sz[i])];
      for (int j = 0; j < batch_size; ++j)
      {
        object_box_cpu[i][7*j] = CPUCompute2StorageT(ComputeT(j));
      }
    }
    noise_value_cpu = new StorageT[numel_batch_noise];

    object_cls_val_gpu.assign(object_num, NULL);
    object_reg_bin_gpu.assign(object_num, NULL);
    object_reg_val_gpu.assign(object_num, NULL);
    object_box_gpu.assign(object_num, NULL);

    checkCUDA(__LINE__, cudaMalloc(&data_gpu, numel_batch_data * sizeofStorageT) );
    for (int i = 0; i < object_cls_sz.size(); ++i)
    {
      checkCUDA(__LINE__, cudaMalloc(&object_cls_val_gpu[i], numel(object_cls_sz[i]) * sizeofStorageT) );
      checkCUDA(__LINE__, cudaMalloc(&object_reg_bin_gpu[i], numel(object_reg_sz[i]) * sizeofStorageT) );
      checkCUDA(__LINE__, cudaMalloc(&object_reg_val_gpu[i], numel(object_reg_sz[i]) * sizeofStorageT) );
      checkCUDA(__LINE__, cudaMalloc(&object_box_gpu[i], numel(object_box_sz[i]) * sizeofStorageT) );
    }
    checkCUDA(__LINE__, cudaMalloc(&noise_value_gpu, numel_batch_noise * sizeofStorageT) );

    id_manager.init( data_list_id, label_list, 8, rng , balance_data, shuffle_data );
    srand (time(NULL));


    //// predefine testing
    if (phase == Testing && use_predef)
    {
      cout << "Pre-define mode..." << endl;

      cout << "Read pre-define anchor: " << predef_anchor_file << endl;
      dataFILE = fopen(predef_anchor_file.c_str(), "rb");
      if (dataFILE==NULL){
        cerr<<"Fail to open the data file"<<endl;
        FatalError(__LINE__);
      }
      unsigned int temp_num_data;
      fread( (void*)(&temp_num_data), sizeof(unsigned int), 1, dataFILE);
      cout << "temp_num_data: " << temp_num_data << endl;
      predef_anchor.assign(temp_num_data, SceneTemplate());
      for (int sid=0; sid<temp_num_data; sid++)
      {
        predef_anchor[sid].loadTemplate(dataFILE);
      }
      fclose(dataFILE);

      cout << "Read pre-define noise..." << endl;
      dataFILE = fopen(predef_noise_file.c_str(), "rb");
      predef_noise = new float[3*predef_anchor.size()];
      if (dataFILE==NULL){
        cerr<<"Fail to open the data file"<<endl;
        FatalError(__LINE__);
      }
      fread((void*)predef_noise, sizeof(float), 3*predef_anchor.size(), dataFILE);
      fclose(dataFILE);
    }

  };

  SceneHolisticOnlineTSDFROILayer(string name_, Phase phase_): DataLayer(name_){
    phase = phase_;
    init();
  };

  SceneHolisticOnlineTSDFROILayer(JSON* json){
    SetValue(json, name,    "Whatever")
    SetValue(json, phase,   Training)
    SetOrDie(json, template_avg_file )
    SetOrDie(json, template_std_file )
    SetOrDie(json, ground_truth_file )
    SetOrDie(json, camera_info_file )
    SetOrDie(json, file_list )
    SetOrDie(json, file_label )
    SetOrDie(json, data_root )
    SetOrDie(json, grid_size )
    SetOrDie(json, spatial_range )
    SetValue(json, batch_size,  64)
    SetValue(json, shuffle_data, true);
    SetValue(json, balance_data, true);
    SetValue(json, noise_scale, vector<float>(3,1.5))
    SetOrDie(json, template_id)
    SetValue(json, use_predef, false);
    SetValue(json, predef_noise_file, "");
    SetValue(json, predef_anchor_file, "");
    SetValue(json, disturb, false);

    cout << "template_avg_file:　" << template_avg_file << endl;
    cout << "template_std_file:　" << template_std_file << endl;
    cout << "ground_truth_file:　" << ground_truth_file << endl;
    cout << "file_list:　" << file_list << endl;
    cout << "file_label: " << file_label << endl;
    cout << "camera_info_file: " << camera_info_file << endl;
    cout << "data_root: " << data_root << endl;
    cout << "grid_size: " << vecPrintString(grid_size) << endl;
    cout << "spatial_range: " << vecPrintString(spatial_range) << endl;
    cout << "batch_size: " << batch_size << endl;
    cout << "noise_scale: " << vecPrintString(noise_scale) << endl;
    init();
  };

  ~SceneHolisticOnlineTSDFROILayer(){
    // if (data_gpu != NULL) checkCUDA(__LINE__, cudaFree(data_gpu));
    // if (bbox_value_gpu != NULL) checkCUDA(__LINE__, cudaFree(bbox_value_gpu));
    // if (bbox_valid_gpu != NULL) checkCUDA(__LINE__, cudaFree(bbox_valid_gpu));
    // if (bbox_weight_gpu != NULL) checkCUDA(__LINE__, cudaFree(bbox_weight_gpu));

    // if (data_cpu) delete [] data_cpu;
    // if (bbox_value_cpu) delete [] bbox_value_cpu;
    // if (bbox_valid_cpu) delete [] bbox_valid_cpu;
    // if (bbox_weight_cpu) delete [] bbox_weight_cpu;
  };


  void shuffle(){
    if (shuffle_data){
      ordering = randperm( data_list_name.size(), rng);
    }
    else {
      ordering.resize(data_list_name.size());
      for (int i = 0; i < data_list_name.size(); ++i ) ordering[i]=i;
    }
  }; 

  void prefetch(){

    checkCUDA(__LINE__,cudaSetDevice(GPU));
    // tic(); cout<<"read disk  ";
    // size_t read_count;

    // for (int i = 0; i < object_num; ++i)
    // {
    //   memset( (void*)object_cls_val_cpu[i], 0, sizeofStorageT * numel(object_cls_sz[i]));
    //   memset( (void*)object_reg_bin_cpu[i], 0, sizeofStorageT * numel(object_reg_sz[i]));
    //   memset( (void*)object_reg_val_cpu[i], 0, sizeofStorageT * numel(object_reg_sz[i]));
    //   memset( (void*)object_box_cpu[i], 0, sizeofStorageT * numel(object_box_sz[i]));
    // }
    // memset( (void*)noise_value_cpu, 0, sizeofStorageT * numel_batch_noise);

    for (size_t i = 0; i < batch_size; ++i)
    {
      StorageT* noise_value_target = noise_value_cpu + i * numel_single_noise;
      std::vector<StorageT*> object_cls_val_target(object_num, NULL);
      std::vector<StorageT*> object_reg_bin_target(object_num, NULL);
      std::vector<StorageT*> object_reg_val_target(object_num, NULL);
      std::vector<StorageT*> object_box_target(object_num, NULL);
      for (int j = 0; j < object_num; ++j)
      {
        object_cls_val_target[j] = object_cls_val_cpu[j] + i * sizeofitem(object_cls_sz[j]);
        object_reg_bin_target[j] = object_reg_bin_cpu[j] + i * sizeofitem(object_reg_sz[j]);
        object_reg_val_target[j] = object_reg_val_cpu[j] + i * sizeofitem(object_reg_sz[j]);
        object_box_target[j] = object_box_cpu[j] + i * sizeofitem(object_box_sz[j]) + 1;
      }

      int local_id = id_manager.next_id(epoch_prefetch);
      // cout << "local_id: " << local_id << endl;

      string data_name = data_list_name[local_id];
      string data_file = data_root + data_name + "_depth.bin";    
      int data_id = data_list_id[local_id];
      // cout << "data_id: " << data_id << ", label: " << label_list[data_id] << endl;

      float noise_dif[3];      
      if ( use_predef && phase == Testing) {
        noise_dif[0] = predef_noise[local_id*3+0];
        noise_dif[1] = predef_noise[local_id*3+1];
        noise_dif[2] = predef_noise[local_id*3+2];
      }
      else {
        noise_dif[0] = 2.0*(float)(rand()%100)/100*noise_scale[0] - noise_scale[0];
        noise_dif[1] = 2.0*(float)(rand()%100)/100*noise_scale[1] - noise_scale[1];
        noise_dif[2] = 2.0*(float)(rand()%100)/100*noise_scale[2] - noise_scale[2];
      }
      // cout << "noise_diff: " << noise_dif[0] << " " << noise_dif[1] << " " << noise_dif[2] << endl;

      float noise_tsl[3];
      noise_tsl[0] = ground_truth[data_id].Tsl[0] + noise_dif[0];
      noise_tsl[1] = ground_truth[data_id].Tsl[1] + noise_dif[1];
      noise_tsl[2] = ground_truth[data_id].Tsl[2] + noise_dif[2];
      for (int j = 0; j<3; j++)
      {
        noise_value_target[j] = CPUCompute2StorageT( ComputeT(noise_dif[j]) );
      }

      depth_image.readImage(data_file);
      depth_image.SetupMatrix(ground_truth[data_id].K, ground_truth[data_id].R, ground_truth[data_id].Rot, noise_tsl);
      depth_image.transformGridPoints();
      depth_image.ComputeTSDF_storage();
      checkCUDA(__LINE__, cudaMemcpy( data_gpu + i*numel_single_data,  depth_image.tsdf_gpu_storage,  numel_single_data * sizeofStorageT, cudaMemcpyDeviceToDevice) );

      SceneTemplate transform_gnd = ground_truth[data_id].gnd;
      // cout << "Original center: " << transform_gnd.objects[0].center[0] << " " << transform_gnd.objects[0].center[1] << endl;

      transform_gnd.transformTemplate(ground_truth[data_id].Rot, noise_tsl);
      // cout << "Original center: " << transform_gnd.objects[0].center[0] << " " << transform_gnd.objects[0].center[1] << endl;

      SceneTemplate disturb_anchor;
      if ( use_predef && phase == Testing)
      {
        disturb_anchor = predef_anchor[local_id];
      }
      else
      {
        disturb_anchor = template_avg[template_id];
        if (disturb)
        {
          disturb_anchor.disturbTemplate(template_avg[template_id], template_std[template_id], rng);
        }       
      }

      disturb_anchor.getIOUbased_value_storage( object_cls_val_target, object_reg_bin_target, object_reg_val_target, 
        object_box_target, transform_gnd, template_std[template_id], spatial_reverse, grid_size_reverse);
      //getIOUbased_value_storage( vector<StorageT*> cls_val, vector<StorageT*> reg_bin, vector<StorageT*> reg_val, vector<StorageT*> &boxes, SceneTemplate &gnd, SceneTemplate &norm, vector<float> spatial_range, float grid_size)
    
      // for (int k = 3; k < 13; ++k)
      // {
      //   cout << k << " " << CPUStorage2ComputeT(object_reg_bin_target[k][1]) << ": ";
      //   for (int j=0; j<6; j++)
      //   {
      //     cout << CPUStorage2ComputeT(object_reg_val_target[k][j]) << " ";
      //   }
      //   cout << endl;
      // }

      // FatalError(__LINE__);
    }
    // FatalError(__LINE__);
    /////////////////// ship to gpu
    for (int i = 0; i < object_num; ++i)
    {
      checkCUDA(__LINE__, cudaMemcpy( object_cls_val_gpu[i], object_cls_val_cpu[i], numel(object_cls_sz[i]) * sizeofStorageT, cudaMemcpyHostToDevice) );
      checkCUDA(__LINE__, cudaMemcpy( object_reg_bin_gpu[i], object_reg_bin_cpu[i], numel(object_reg_sz[i]) * sizeofStorageT, cudaMemcpyHostToDevice) );
      checkCUDA(__LINE__, cudaMemcpy( object_reg_val_gpu[i], object_reg_val_cpu[i], numel(object_reg_sz[i]) * sizeofStorageT, cudaMemcpyHostToDevice) );
      checkCUDA(__LINE__, cudaMemcpy( object_box_gpu[i], object_box_cpu[i], numel(object_box_sz[i]) * sizeofStorageT, cudaMemcpyHostToDevice) );
    }
    checkCUDA(__LINE__, cudaMemcpy( noise_value_gpu, noise_value_cpu, numel_batch_noise * sizeofStorageT, cudaMemcpyHostToDevice) );
  };

  void forward(Phase phase_){
    lock.wait();
    epoch = epoch_prefetch;
    
    swap( out[0]->dataGPU, data_gpu);
    swap( out[1]->dataGPU, noise_value_gpu);
    for (int i = 0; i < object_num; ++i)
    {
      swap( out[2+i*4+0]->dataGPU, object_cls_val_gpu[i]);
      swap( out[2+i*4+1]->dataGPU, object_reg_bin_gpu[i]);
      swap( out[2+i*4+2]->dataGPU, object_reg_val_gpu[i]);
      swap( out[2+i*4+3]->dataGPU, object_box_gpu[i]);
    }

    lock = async( launch::async, &SceneHolisticOnlineTSDFROILayer::prefetch, this);
  };


  size_t Malloc(Phase phase_){

    if (phase == Training && phase_==Testing) return 0;

    size_t memoryBytes = 0;

    // cout<< (train_me? "* " : "  ");
    // cout<<name<<endl;

    cout << "#Output: " << out.size() << endl;

    out[0]->need_diff = false;
    memoryBytes += 2 * out[0]->Malloc(data_dims);
    cout << "MemoryBytes: " << memoryBytes << endl;

    out[1]->need_diff = false;
    memoryBytes += 2 * out[1]->Malloc(noise_value_dims);
    cout << "MemoryBytes: " << memoryBytes << endl;

    for (int i = 0; i < object_num; ++i)
    {
      out[2+i*4+0]->need_diff = false;
      memoryBytes += 2 * out[2+i*4+0]->Malloc(object_cls_sz[i]);
      cout << "MemoryBytes: " << memoryBytes << endl; 
      out[2+i*4+1]->need_diff = false;
      memoryBytes += 2 * out[2+i*4+1]->Malloc(object_reg_sz[i]);
      cout << "MemoryBytes: " << memoryBytes << endl; 
      out[2+i*4+2]->need_diff = false;
      memoryBytes += 2 * out[2+i*4+2]->Malloc(object_reg_sz[i]);
      cout << "MemoryBytes: " << memoryBytes << endl; 
      out[2+i*4+3]->need_diff = false;
      memoryBytes += 2 * out[2+i*4+3]->Malloc(object_box_sz[i]);
      cout << "MemoryBytes: " << memoryBytes << endl; 
    }

    cout << "Memory allocated, read the first batch of data..." << endl;
    lock = async( launch::async, &SceneHolisticOnlineTSDFROILayer::prefetch, this);

    return memoryBytes;
    // return 0;
  };  
};


 