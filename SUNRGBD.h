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
using namespace std;


// template <typename T>
// string vecPrintString(vector<T> &v)
// {
//   stringstream convert;
//   convert << "[" << v.size() << "]={";
//   if (v.size()>0)  convert << v[0];
//   if (v.size()>1){  
//     for (int i = 1; i < v.size(); ++i){
//       convert << "," << v[i];
//     }
//   }
//   convert << "}"; //<<endl;
//   string result = convert.str();
//   return result;
// }

// struct Point3D {
//   float x, y, z;
//   Point3D(float xi, float yi, float zi): x(xi), y(yi), z(zi) {};
//   Point3D(): x(0), y(0), z(0) {};
// };

// __global__ void compute_xyzkernel(float* XYZimage, float* depthMap, float * K, float * R){
//   int iy = blockIdx.x;
//   int ix = threadIdx.x;
//   int width = blockDim.x;
 
//   // printf("%d\n", width);
//   int index = iy * width + ix;

//   float depth = depthMap[ index ];

//   if (depth < 0.00001)
//   {
//     XYZimage[3 * index + 0] = 0;
//     XYZimage[3 * index + 1] = 0;
//     XYZimage[3 * index + 2] = 0;
//   }
//   else
//   {
//     // project the depth point to 3d
//     float tdx = (float(ix + 1) - K[2]) * depth / K[0];
//     float tdz =  - (float(iy + 1) - K[5]) * depth / K[4];
//     float tdy = depth;

//     XYZimage[3 * index + 0] = R[0] * tdx + R[1] * tdy + R[2] * tdz;
//     XYZimage[3 * index + 1] = R[3] * tdx + R[4] * tdy + R[5] * tdz;
//     XYZimage[3 * index + 2] = R[6] * tdx + R[7] * tdy + R[8] * tdz;
//     // XYZimage[3 * (iy*width + ix) + 0] = tdx;
//     // XYZimage[3 * (iy*width + ix) + 1] = tdy;
//     // XYZimage[3 * (iy*width + ix) + 2] = tdz;  
//   }
  
  
// }

// __global__ void compute_TSDFGPUbox_proj(float* tsdf_data, float* R_data, float* K_data, float* depth, float* XYZimage,
//                                       const float* scene_setting, int im_w, int im_h)
// {
//     const int index = threadIdx.x + blockIdx.x * blockDim.x;
//     int tsdf_size[3];
//     tsdf_size[0] = scene_setting[6];
//     tsdf_size[1] = scene_setting[7];
//     tsdf_size[2] = scene_setting[8];

//     int volume_size = tsdf_size[0] * tsdf_size[1] * tsdf_size[2];
//     if (index > volume_size) return;
//     float delta_x = (scene_setting[1]-scene_setting[0]) / float(scene_setting[6]);  
//     float delta_y = (scene_setting[3]-scene_setting[2]) / float(scene_setting[7]);   
//     float delta_z = (scene_setting[5]-scene_setting[4]) / float(scene_setting[8]);

//     float truncated_value = scene_setting[9];
//     float reg_scale = 1.0/truncated_value/truncated_value;
//     float num_channel = 3;


//     float x = float((index / ( tsdf_size[1] * tsdf_size[2] ) ) % tsdf_size[0]) ;
//     float y = float((index / tsdf_size[2]) % tsdf_size[1] );
//     float z = float(index % tsdf_size[2]);

//     // for (int i =0;i<num_channel;i++){
//     //     tsdf_data[index + i * volume_size] = 0;
//     // }

//     // get grid world coordinate
//     float temp_x = scene_setting[0] + (x + 0.5) * delta_x;
//     float temp_y = scene_setting[2] + (y + 0.5) * delta_y;
//     float temp_z = scene_setting[4] + (z + 0.5) * delta_z;

//     x = temp_x;
//     y = temp_y;
//     z = temp_z;

//     // x = temp_x * bb3d_data[0] + temp_y * bb3d_data[3] + temp_z * bb3d_data[6]
//     //     + bb3d_data[9];
//     // y = temp_x * bb3d_data[1] + temp_y * bb3d_data[4] + temp_z * bb3d_data[7]
//     //     + bb3d_data[10];
//     // z = temp_x * bb3d_data[2] + temp_y * bb3d_data[5] + temp_z * bb3d_data[8]
//     //     + bb3d_data[11]; 

//     // project to image plane decides the sign
//     // rotate back and swap y, z and -y
//     float xx =   R_data[0] * x + R_data[3] * y + R_data[6] * z;
//     float zz =   R_data[1] * x + R_data[4] * y + R_data[7] * z;
//     float yy = - R_data[2] * x - R_data[5] * y - R_data[8] * z;
//     int ix = floor(xx * K_data[0] / zz + K_data[2]+0.5) - 1;
//     int iy = floor(yy * K_data[4] / zz + K_data[5]+0.5) - 1;

//     // 1. out of image case
//     if (ix < 0 || ix >= im_w || iy < 0 || iy >= im_h || zz < 0.0001) {
//       tsdf_data[index] = 0; // value
//       tsdf_data[index + volume_size] = 0; // No depth
//       tsdf_data[index + 2*volume_size] = -1; // Out of image
//       // tsdf_data[index + 3*volume_size] = 0; // undefined front back
//       return;  
//     }
//     // 2. missing depth case
//     int img_index = iy * im_w + ix;
//     float depth_onsite = depth[img_index];
//     if ( depth_onsite < 0.0001 || depth_onsite > -0.0001 ) {
//       tsdf_data[index] = 0; // value
//       tsdf_data[index + volume_size] = 0; // No depth
//       tsdf_data[index + 2*volume_size] = 1; // Inside image
//       // tsdf_data[index + 3*volume_size] = 0; // undefined front back
//       return;
//     }

//     // 3. normal case, some 3D points
//     float x_project   = XYZimage[3*img_index+0];
//     float y_project   = XYZimage[3*img_index+1];
//     float z_project   = XYZimage[3*img_index+2]; 

//     float tsdf_x = abs(x - x_project);
//     float tsdf_y = abs(y - y_project);
//     float tsdf_z = abs(z - z_project);

//     float dist_to_surface_square = tsdf_x * tsdf_x + tsdf_y * tsdf_y + tsdf_z * tsdf_z;
//     float tsdf_value = max(1 - reg_scale*dist_to_surface_square, 0.0);
//     tsdf_data[index] = (zz > y_project) ? -tsdf_value:tsdf_value; // value
//     tsdf_data[index + volume_size] = 1; // Has depth
//     tsdf_data[index + 2*volume_size] = 1; // Inside image
// }

// __global__ void compute_TSDFGPU_grid_proj(float* tsdf_data, float* R_data, float* K_data, float* depth, float* XYZimage,
//                                       const float* grid, int grid_size, float truncated_value, int im_w, int im_h)
// {
//     const int index = threadIdx.x + blockIdx.x * blockDim.x;
//     int volume_size = grid_size;
//     if (index >= volume_size) return;

//     float reg_scale = 1.0/truncated_value/truncated_value;
//     float num_channel = 3;

//     // printf("%f\n", reg_scale);

//     float x = grid[ index*3 ];
//     float y = grid[ index*3 + 1 ];
//     float z = grid[ index*3 + 2 ];

//     // for (int i =0;i<num_channel;i++){
//     //     tsdf_data[index + i * volume_size] = 0;
//     // }

//     // get grid world coordinate
//     // float temp_x = scene_setting[0] + (x + 0.5) * delta_x;
//     // float temp_y = scene_setting[2] + (y + 0.5) * delta_y;
//     // float temp_z = scene_setting[4] + (z + 0.5) * delta_z;

//     // x = temp_x;
//     // y = temp_y;
//     // z = temp_z;

//     // x = temp_x * bb3d_data[0] + temp_y * bb3d_data[3] + temp_z * bb3d_data[6]
//     //     + bb3d_data[9];
//     // y = temp_x * bb3d_data[1] + temp_y * bb3d_data[4] + temp_z * bb3d_data[7]
//     //     + bb3d_data[10];
//     // z = temp_x * bb3d_data[2] + temp_y * bb3d_data[5] + temp_z * bb3d_data[8]
//     //     + bb3d_data[11]; 

//     // project to image plane decides the sign
//     // rotate back and swap y, z and -y
//     float xx =   R_data[0] * x + R_data[3] * y + R_data[6] * z;
//     float zz =   R_data[1] * x + R_data[4] * y + R_data[7] * z;
//     float yy = - R_data[2] * x - R_data[5] * y - R_data[8] * z;
//     int ix = floor(xx * K_data[0] / zz + K_data[2]+0.5) - 1;
//     int iy = floor(yy * K_data[4] / zz + K_data[5]+0.5) - 1;

//     // 1. out of image case
//     if (ix < 0 || ix >= im_w || iy < 0 || iy >= im_h || zz < 0.0001) {
//       tsdf_data[index] = 0; // value
//       tsdf_data[index + volume_size] = 0; // No depth
//       tsdf_data[index + 2*volume_size] = -1; // Out of image
//       // tsdf_data[index + 3*volume_size] = 0; // undefined front back
//       return;  
//     }
//     // 2. missing depth case
//     int img_index = iy * im_w + ix;
//     float depth_onsite = depth[img_index];
//     if ( depth_onsite < 0.0001 ) {
//       tsdf_data[index] = 0; // value
//       tsdf_data[index + volume_size] = 0; // No depth
//       tsdf_data[index + 2*volume_size] = 1; // Inside image
//       // tsdf_data[index + 3*volume_size] = 0; // undefined front back
//       return;
//     }
//     // printf("%d, %d, %f\n", ix, iy, depth_onsite);
//     // 3. normal case, some 3D points
//     float x_project   = XYZimage[3*img_index+0];
//     float y_project   = XYZimage[3*img_index+1];
//     float z_project   = XYZimage[3*img_index+2]; 

//     float tsdf_x = abs(x - x_project);
//     float tsdf_y = abs(y - y_project);
//     float tsdf_z = abs(z - z_project);

//     float dist_to_surface_square = tsdf_x * tsdf_x + tsdf_y * tsdf_y + tsdf_z * tsdf_z;
//     float tsdf_value = max(1 - reg_scale*dist_to_surface_square, 0.0);
//     tsdf_data[index] = (zz > y_project) ? -tsdf_value:tsdf_value; // value
//     tsdf_data[index + volume_size] = 1; // Has depth
//     tsdf_data[index + 2*volume_size] = 1; // Inside image

//     // if (index==1)
//     // {
//     //   printf("depth: %f\n", depth_onsite);
//     //   printf("img_index: %d\n", img_index);
//     //   printf("ix: %d, iy: %d\n", ix, iy); 
//     //   printf("x_p: %f, y_p: %f, z_p: %f\n", x_project, y_project, z_project);
//     //   printf("x: %f, y: %f, z: %f\n", x, y, z);
//     // }
// }

// __global__ void compute_TSDFGPU_grid_proj_storageT(StorageT* tsdf_data, float* R_data, float* K_data, float* depth, float* XYZimage,
//                                       const float* grid, int grid_size, float truncated_value, int im_w, int im_h)
// {
//     const int index = threadIdx.x + blockIdx.x * blockDim.x;
//     int volume_size = grid_size;
//     if (index >= volume_size) return;

//     float reg_scale = 1.0/truncated_value/truncated_value;
//     float num_channel = 3;

//     // printf("%f\n", reg_scale);

//     float x = grid[ index*3 ];
//     float y = grid[ index*3 + 1 ];
//     float z = grid[ index*3 + 2 ];

//     // for (int i =0;i<num_channel;i++){
//     //     tsdf_data[index + i * volume_size] = 0;
//     // }

//     // get grid world coordinate
//     // float temp_x = scene_setting[0] + (x + 0.5) * delta_x;
//     // float temp_y = scene_setting[2] + (y + 0.5) * delta_y;
//     // float temp_z = scene_setting[4] + (z + 0.5) * delta_z;

//     // x = temp_x;
//     // y = temp_y;
//     // z = temp_z;

//     // x = temp_x * bb3d_data[0] + temp_y * bb3d_data[3] + temp_z * bb3d_data[6]
//     //     + bb3d_data[9];
//     // y = temp_x * bb3d_data[1] + temp_y * bb3d_data[4] + temp_z * bb3d_data[7]
//     //     + bb3d_data[10];
//     // z = temp_x * bb3d_data[2] + temp_y * bb3d_data[5] + temp_z * bb3d_data[8]
//     //     + bb3d_data[11]; 

//     // project to image plane decides the sign
//     // rotate back and swap y, z and -y
//     float xx =   R_data[0] * x + R_data[3] * y + R_data[6] * z;
//     float zz =   R_data[1] * x + R_data[4] * y + R_data[7] * z;
//     float yy = - R_data[2] * x - R_data[5] * y - R_data[8] * z;
//     int ix = floor(xx * K_data[0] / zz + K_data[2]+0.5) - 1;
//     int iy = floor(yy * K_data[4] / zz + K_data[5]+0.5) - 1;

//     // 1. out of image case
//     if (ix < 0 || ix >= im_w || iy < 0 || iy >= im_h || zz < 0.0001) {
//       tsdf_data[index] = GPUCompute2StorageT( ComputeT(0) ); // value
//       tsdf_data[index + volume_size] = GPUCompute2StorageT( ComputeT(0) ); // No depth
//       tsdf_data[index + 2*volume_size] = GPUCompute2StorageT( ComputeT(-1) ); // Out of image
//       // tsdf_data[index + 3*volume_size] = 0; // undefined front back
//       return;  
//     }
//     // 2. missing depth case
//     int img_index = iy * im_w + ix;
//     float depth_onsite = depth[img_index];
//     if ( depth_onsite < 0.0001 ) {
//       tsdf_data[index] = GPUCompute2StorageT( ComputeT(0) ); // value
//       tsdf_data[index + volume_size] = GPUCompute2StorageT( ComputeT(0) ); // No depth
//       tsdf_data[index + 2*volume_size] = GPUCompute2StorageT( ComputeT(1) ); // Inside image
//       // tsdf_data[index + 3*volume_size] = 0; // undefined front back
//       return;
//     }
//     // printf("%d, %d, %f\n", ix, iy, depth_onsite);
//     // 3. normal case, some 3D points
//     float x_project   = XYZimage[3*img_index+0];
//     float y_project   = XYZimage[3*img_index+1];
//     float z_project   = XYZimage[3*img_index+2]; 

//     float tsdf_x = abs(x - x_project);
//     float tsdf_y = abs(y - y_project);
//     float tsdf_z = abs(z - z_project);

//     float dist_to_surface_square = tsdf_x * tsdf_x + tsdf_y * tsdf_y + tsdf_z * tsdf_z;
//     float tsdf_value = max(1 - reg_scale*dist_to_surface_square, 0.0);
//     float temp = (zz > y_project) ? -tsdf_value:tsdf_value;
//     tsdf_data[index] =  GPUCompute2StorageT( ComputeT(temp) );// value
//     tsdf_data[index + volume_size] = GPUCompute2StorageT( ComputeT(1) ); // Has depth
//     tsdf_data[index + 2*volume_size] = GPUCompute2StorageT( ComputeT(1) ); // Inside image

//     // if (index==1)
//     // {
//     //   printf("depth: %f\n", depth_onsite);
//     //   printf("img_index: %d\n", img_index);
//     //   printf("ix: %d, iy: %d\n", ix, iy); 
//     //   printf("x_p: %f, y_p: %f, z_p: %f\n", x_project, y_project, z_project);
//     //   printf("x: %f, y: %f, z: %f\n", x, y, z);
//     // }
// }

// __global__ void compute_grid(float* transform_grid_gpu, float* grid_gpu, float* Rot_GPU, float* Tsl_GPU, int grid_size)
// {
//   const int index = threadIdx.x + blockIdx.x * blockDim.x;
  
  
//   // printf("%d: %d %d %d\n", index, threadIdx.x, blockIdx.x, blockDim.x);

//   if (index >= grid_size) return;

//   int xid = 3 * index;
//   // printf("grid: %d, index: %d, xid: %d\n", grid_size, index, xid);
//   // printf("R: %f %f %f %f, T: %f %f %f;", Rot_GPU[0], Rot_GPU[1], Rot_GPU[2], Rot_GPU[3], Tsl_GPU[0], Tsl_GPU[1], Tsl_GPU[2]);
//   float x = grid_gpu[xid];
//   float y = grid_gpu[xid + 1];
//   float z = grid_gpu[xid + 2];

//   transform_grid_gpu[xid] = Rot_GPU[0]*x + Rot_GPU[1]*y - Tsl_GPU[0];
//   transform_grid_gpu[xid+1] = Rot_GPU[2]*x + Rot_GPU[3]*y - Tsl_GPU[1];
//   transform_grid_gpu[xid+2] = z - Tsl_GPU[2];

//   // printf(" %d: %f %f %f, %f %f %f\n", index, x, y, z, transform_grid_gpu[xid], transform_grid_gpu[xid+1], transform_grid_gpu[xid+2]);
// }

// struct Box3D{
//   unsigned int category;
//   float orientation[3];
//   float center[3];
//   float coeff[3];

//   Box3D(): category(-1) {};

//   Box3D(const Box3D &b)
//   {
//     category = b.category;
//     orientation[0] = b.orientation[0];
//     orientation[1] = b.orientation[1];
//     orientation[2] = b.orientation[2];
//     center[0] = b.center[0];
//     center[1] = b.center[1];
//     center[2] = b.center[2];
//     coeff[0] = b.coeff[0];
//     coeff[1] = b.coeff[1];
//     coeff[2] = b.coeff[2];
//   }

//   void transformBox3D(float* Rot, float* Tsl)
//   {
//     float ox = orientation[0];
//     float oy = orientation[1];
//     orientation[0] = Rot[0]*ox + Rot[2]*oy;
//     orientation[1] = Rot[1]*ox + Rot[3]*oy;

//     // float cx = center[0];
//     // float cy = center[1];
//     // float cz = center[2];
//     // center[0] = Rot[0]*cx + Rot[2]*cy + Tsl[0];
//     // center[1] = Rot[1]*cx + Rot[3]*cy + Tsl[1];
//     // center[2] = cz + Tsl[2];
//     float cx = center[0] + Tsl[0];
//     float cy = center[1] + Tsl[1];
//     float cz = center[2] + Tsl[2];
//     center[0] = Rot[0]*cx + Rot[2]*cy;
//     center[1] = Rot[1]*cx + Rot[3]*cy;
//     center[2] = cz;
//   }

//   void load(FILE* &fp)
//   {
//     fread((void*)(&category), sizeof(unsigned int), 1, fp);
//     // cout << "category: " << category << endl;
//     fread((void*)center, sizeof(unsigned int), 3, fp);
//     // cout << "center: " << center[0] << " " << center[1] << " " << center[2] << endl;
//     fread((void*)coeff, sizeof(unsigned int), 3, fp);
//     // cout << "coeff: " << coeff[0] << " " << coeff[1] << " " << coeff[2] << endl;
//     fread((void*)orientation, sizeof(unsigned int), 3, fp);
//     // cout << "orientation: " << orientation[0] << " " << orientation[1] << " " << orientation[2] << endl;
//   }
// };

// // enum WallType { isa_floor; isa_ceiling; isa_wall;}

// struct Wall3D {
//   unsigned int type;
//   float top_loc[2];
//   // vector<float> points;
//   // float* points;
//   // unsigned int num_points;
//   Wall3D() {
//     type = 0;
//     // points.clear();
//     // num_points = 0;
//   }

//   Wall3D(const Wall3D &w)
//   {
//     type = w.type;
//     top_loc[0] = w.top_loc[0];
//     top_loc[1] = w.top_loc[1];
//   }

//   void transformWall3D(float* Rot, float* Tsl)
//   {
//     if (type==1 || type==2)
//     {
//       top_loc[0] = top_loc[0] + Tsl[2];
//     }
//     else if (type==0)
//     {
//       // float p1[2];
//       // float p2[2];
//       // p1[0] = top_loc[0];
//       // p1[1] = top_loc[1];
//       // p2[0] = top_loc[0] + top_loc[1];
//       // p2[1] = top_loc[1] - top_loc[0];

//       // float tp1[2];
//       // float tp2[2];
//       // tp1[0] = Rot[0]*p1[0] + Rot[2]*p1[1] + Tsl[0];
//       // tp1[1] = Rot[1]*p1[0] + Rot[3]*p1[1] + Tsl[1];
//       // tp2[0] = Rot[0]*p2[0] + Rot[2]*p2[1] + Tsl[0];
//       // tp2[1] = Rot[1]*p2[0] + Rot[3]*p2[1] + Tsl[1];

//       // cout << "top_loc: " << top_loc[0] << " " << top_loc[1] << endl;
//       float p1[2];
//       float p2[2];
//       p1[0] = top_loc[0] + Tsl[0];
//       p1[1] = top_loc[1] + Tsl[1];
//       p2[0] = top_loc[0] + top_loc[1] + Tsl[0];
//       p2[1] = top_loc[1] - top_loc[0] + Tsl[1];

//       // cout << "Tsl: " << Tsl[0] << " " << Tsl[1] << " " << Tsl[2] << endl;
//       // cout << "p1: " << p1[0] << " " << p1[1] << endl;
//       // cout << "p2: " << p2[0] << " " << p2[1] << endl;

//       float tp1[2];
//       float tp2[2];
//       tp1[0] = Rot[0]*p1[0] + Rot[2]*p1[1];
//       tp1[1] = Rot[1]*p1[0] + Rot[3]*p1[1];
//       tp2[0] = Rot[0]*p2[0] + Rot[2]*p2[1];
//       tp2[1] = Rot[1]*p2[0] + Rot[3]*p2[1];

//       float k_line[2];
//       k_line[0] = tp2[1] - tp1[1];
//       k_line[1] = tp1[0] - tp2[0];
//       float temp = sqrt(k_line[0]*k_line[0] + k_line[1]*k_line[1]);
//       k_line[0] /= temp;
//       k_line[1] /= temp;
//       float dist = tp1[0]*k_line[0] + tp1[1]*k_line[1];
//       top_loc[0] = dist * k_line[0];
//       top_loc[1] = dist * k_line[1]; 
//       // cout << "top_loc: " << top_loc[0] << " " << top_loc[1] << endl;
//     }
//   }

//   void load(FILE* &fp)
//   {
//     fread((void*)(&(type)), sizeof(unsigned int), 1, fp);
//     // cout << "type: " << type << endl;
//     fread((void*)top_loc, sizeof(float), 2, fp);
//     // cout << "top_loc: " << top_loc[0] << ", " << top_loc[1] << endl;
//   }

//   // void load(FILE* &fp)
//   // {
//   //   fread((void*)(&(type)), sizeof(unsigned int), 1, fp);
//   //   cout << "type: " << type << endl;
//   //   fread((void*)(&(num_points)), sizeof(unsigned int), 1, fp);
//   //   cout << "num_points: " << num_points << endl;
//   //   float *p = new float[num_points*3];
//   //   fread((void*)p, sizeof(float), num_points*3, fp);
//   //   for (int i = 0; i<num_points*3; i++)
//   //   {
//   //     points.push_back(p[i]);
//   //     cout << points[i] << " ";
//   //   }
//   //   cout << endl;
//   //   delete[] p;
//   // }
//   ~Wall3D()
//   {
//   }

// };

// struct DepthImage {
//   float* depth_cpu;
//   // float* scene_setting_cpu;
//   // float* K_CPU;
//   // float* R_CPU;
//   float* K_GPU;
//   float* R_GPU;
//   float* Rot_GPU;
//   float* Tsl_GPU;
//   unsigned int width;
//   unsigned int height;
//   unsigned int memorysize;

//   float* tsdf_gpu;
//   StorageT* tsdf_gpu_storage;

//   float* scene_setting_gpu;
//   float* depth_gpu;
//   float* XYZimage_gpu;

//   float* regular_grid_gpu;
//   float* transform_grid_gpu;
//   int grid_size;
//   int max_grid_size;
//   float truncated_value;

//   DepthImage(): K_GPU(NULL), R_GPU(NULL), Rot_GPU(NULL), Tsl_GPU(NULL), depth_cpu(NULL), width(0), height(0), memorysize(0), tsdf_gpu(NULL), scene_setting_gpu(NULL), depth_gpu(NULL), XYZimage_gpu(NULL), regular_grid_gpu(NULL), transform_grid_gpu(NULL), max_grid_size(0), tsdf_gpu_storage(NULL){
//     // K_CPU = new float[9];
//     // R_CPU = new float[9];
//     // checkCUDA(__LINE__, cudaMalloc(&K_GPU, sizeof(float)*9));
//     // checkCUDA(__LINE__, cudaMalloc(&R_GPU, sizeof(float)*9));
//     // checkCUDA(__LINE__, cudaMalloc(&Rot_GPU, sizeof(float)*4));
//     // checkCUDA(__LINE__, cudaMalloc(&Tsl_GPU, sizeof(float)*3));
//   };

//   void reallocateMemory()
//   {
//     if (memorysize < height*width)
//     {
//       // cout << depth_gpu << " " << depth_cpu << endl;
//       if (depth_cpu != NULL)  delete[] depth_cpu;
//       depth_cpu = new float[ height * width ];
//       if (depth_gpu != NULL)  checkCUDA(__LINE__, cudaFree(depth_gpu)); 
//       checkCUDA(__LINE__, cudaMalloc(&depth_gpu, sizeof(float)*width*height));
//       if (XYZimage_gpu != NULL)  checkCUDA(__LINE__, cudaFree(XYZimage_gpu)); 
//       checkCUDA(__LINE__, cudaMalloc(&XYZimage_gpu, sizeof(float)*width*height*3));
//       memorysize = height * width;
//       // cout << "Reallocate to: " << height << " * " << width << endl;
//       // cout << depth_gpu << " " << depth_cpu << endl;
//     }
//   }

//   void SceneSetting( float* p)
//   {
//     // if (scene_setting_cpu != NULL) scene_setting_cpu = new float[10];
//     if (scene_setting_gpu == NULL) checkCUDA(__LINE__, cudaMalloc(&scene_setting_gpu, sizeof(float)*10));
//     checkCUDA(__LINE__, cudaMemcpy(scene_setting_gpu, p, sizeof(float)*10, cudaMemcpyHostToDevice)); 

//     int x_size = std::round(p[6]);
//     int y_size = std::round(p[7]);
//     int z_size = std::round(p[8]);
//     float x_start = p[0];
//     float x_end = p[1];
//     float y_start = p[2];
//     float y_end = p[3];
//     float z_start = p[4];
//     float z_end = p[5];
//     float x_delta = (x_end-x_start)/x_size;
//     float y_delta = (y_end-y_start)/y_size;
//     float z_delta = (z_end-z_start)/z_size;

//     grid_size = x_size * y_size * z_size;
//     // cout << "x_size: " << x_size << endl;
//     // cout << "y_size: " << y_size << endl;
//     // cout << "z_size: " << z_size << endl;
//     truncated_value = p[9];


//     if (grid_size > max_grid_size)
//     {
//       // cout << "Allocate memory..." << endl;
//       if (tsdf_gpu != NULL) checkCUDA( __LINE__, cudaFree(tsdf_gpu));
//       checkCUDA(__LINE__, cudaMalloc( &tsdf_gpu, sizeof(float) * grid_size * 3));
//       if (tsdf_gpu_storage != NULL) checkCUDA( __LINE__, cudaFree(tsdf_gpu_storage));
//       checkCUDA(__LINE__, cudaMalloc( &tsdf_gpu_storage, sizeofStorageT * grid_size * 3));
//       if (regular_grid_gpu != NULL) checkCUDA( __LINE__, cudaFree(regular_grid_gpu));
//       checkCUDA(__LINE__, cudaMalloc( &regular_grid_gpu, sizeof(float) * grid_size * 3));
//       if (transform_grid_gpu != NULL) checkCUDA( __LINE__, cudaFree(transform_grid_gpu));
//       checkCUDA(__LINE__, cudaMalloc( &transform_grid_gpu, sizeof(float) * grid_size * 3));
//       max_grid_size = grid_size;
//       // cout << "max_grid_size: " << max_grid_size << endl;

//       // cout << "SceneSetting: " << endl;
//       // cout << "tsdf_gpu: " << tsdf_gpu << endl;
//       // cout << "regular_grid_gpu: " << regular_grid_gpu << endl;
//       // cout << "transform_grid_gpu: " << transform_grid_gpu << endl;
//     }  

//     float* grid_cpu = new float[grid_size*3];
//     int count = 0;
//     for (int z = 0; z < z_size; z++)
//       for (int y = 0; y < y_size; y++)
//         for (int x = 0; x < x_size; x++)
//         {
//           grid_cpu[count*3+0] = x_start + ((float)x+0.5) * x_delta;
//           grid_cpu[count*3+1] = y_start + ((float)y+0.5) * y_delta;
//           grid_cpu[count*3+2] = z_start + ((float)z+0.5) * z_delta;
//           count++;
//         }  
//     // cout << "Count: " << count << endl;
//     checkCUDA(__LINE__, cudaMemcpy(regular_grid_gpu, grid_cpu, sizeof(float)*grid_size*3, cudaMemcpyHostToDevice)); 
//     delete[] grid_cpu;
//   }

//   void transformGridPoints()
//   {
//     // checkCUDA(__LINE__, cudaMemcpy(depth_gpu, depth_cpu, sizeof(float)*width*height, cudaMemcpyHostToDevice)); 

//     int THREADS_NUM = 1024;
//     int BLOCK_NUM = int((grid_size + size_t(THREADS_NUM) - 1) / THREADS_NUM);
//     // cout << "BLOCK_NUM: " << BLOCK_NUM << ", THREADS_NUM: " << THREADS_NUM << endl;

//     // cout << "grid_size: " << grid_size << endl;
//     // float* p = new float[grid_size*3];   
//     // FILE* fp = fopen("debug1.bin", "wb");

//     // checkCUDA(__LINE__, cudaMemcpy(p, Rot_GPU, sizeof(float)*4, cudaMemcpyDeviceToHost));
//     // fwrite(p, sizeof(float), 4, fp);
//     // checkCUDA(__LINE__, cudaMemcpy(p, Tsl_GPU, sizeof(float)*3, cudaMemcpyDeviceToHost));
//     // fwrite(p, sizeof(float), 3, fp);

//     // checkCUDA(__LINE__, cudaMemcpy(p, regular_grid_gpu, sizeof(float)*grid_size*3, cudaMemcpyDeviceToHost));
//     // fwrite(p, sizeof(float), grid_size*3, fp);
//     // cout << "Start gpu function" << endl;

//     // cout << "transfore: " << transform_grid_gpu << ", regular: " << regular_grid_gpu << endl;
//     // cout << "Rot_GPU: " << Rot_GPU << ", Tsl_GPU: " << Tsl_GPU << endl;
//     // checkCUDA(__LINE__,cudaDeviceSynchronize());
//     // if (regular_grid_gpu != NULL) checkCUDA(__LINE__, cudaFree(regular_grid_gpu));
//     // checkCUDA(__LINE__, cudaMalloc(&regular_grid_gpu, sizeof(float)*grid_size*3));
//     // if (transform_grid_gpu != NULL) checkCUDA(__LINE__, cudaFree(transform_grid_gpu));
//     // checkCUDA(__LINE__, cudaMalloc(&transform_grid_gpu, sizeof(float)*grid_size*3));
//     // if (Rot_GPU != NULL) checkCUDA(__LINE__, cudaFree(Rot_GPU));
//     // checkCUDA(__LINE__, cudaMalloc(&Rot_GPU, sizeof(float)*4));
//     // if (Tsl_GPU != NULL) checkCUDA(__LINE__, cudaFree(Tsl_GPU));
//     // checkCUDA(__LINE__, cudaMalloc(&Tsl_GPU, sizeof(float)*3));
//     // cout << "new new new" << endl;
//     // cout << "transfore: " << transform_grid_gpu << ", regular: " << regular_grid_gpu << endl;
//     // cout << "Rot_GPU: " << Rot_GPU << ", Tsl_GPU: " << Tsl_GPU << endl;

//     // checkCUDA(__LINE__, cudaMemcpy(p, regular_grid_gpu, sizeof(float)*grid_size*3, cudaMemcpyDeviceToHost));
//     // fwrite(p, sizeof(float), grid_size*3, fp);

//     // checkCUDA(__LINE__,cudaDeviceSynchronize());
//     compute_grid<<<BLOCK_NUM,THREADS_NUM>>>(transform_grid_gpu, regular_grid_gpu, Rot_GPU, Tsl_GPU, grid_size);
//     // checkCUDA(__LINE__, cudaMemcpy(transform_grid_gpu, regular_grid_gpu, sizeof(float)*grid_size*3, cudaMemcpyDeviceToDevice));
//     // checkCUDA(__LINE__,cudaDeviceSynchronize());

//     // cout << "Copy memory back to cpu" << endl;
//     // checkCUDA(__LINE__, cudaMemcpy(p, transform_grid_gpu, sizeof(float)*grid_size*3, cudaMemcpyDeviceToHost));


//     // cout << "Writing to file" << endl;
//     // fwrite(p, sizeof(float), grid_size*3, fp);
//     // delete[] p;
//     // fclose(fp);

//     // checkCUDA(__LINE__, cudaMemcpy(depth_gpu, depth_cpu, sizeof(float)*width*height, cudaMemcpyHostToDevice)); 
//   }

//   void readImage(string &filename)
//   {
//     // cout << "Reading: " << filename << endl;
//       FILE* fp = fopen(filename.c_str(), "rb");
//       // if ( fp == NULL)
//       // {
//       //   cerr<<"Fail to read properly"<<endl;
//       //   FatalError(__LINE__);
//       // }
//       int iBuff;
//       iBuff = fread(&height, sizeof(unsigned int), 1, fp);
//       // cout << "height: " << height << endl;
//       // cout << "iBuff: " << iBuff << endl;
//       if (iBuff != 1) 
//       {
//         cerr<<"Fail to read properly"<<endl;
//         FatalError(__LINE__);
//       }
//       iBuff = fread(&width, sizeof(unsigned int), 1, fp);
//       // cout << "width: " << width << endl;
//       // cout << "iBuff: " << iBuff << endl;
//       if (iBuff != 1) 
//       {
//         cerr<<"Fail to read properly"<<endl;
//         FatalError(__LINE__);
//       }

//       reallocateMemory();

//       iBuff = fread( (void*)depth_cpu, sizeof(float), width*height, fp);
//       // cout << "iBuff: " << iBuff << endl;
//       if (iBuff != width*height) 
//       {
//         cerr<<"Fail to read properly"<<endl;
//         FatalError(__LINE__);
//       }
//       fclose(fp);

//       checkCUDA(__LINE__, cudaMemcpy(depth_gpu, depth_cpu, sizeof(float)*width*height, cudaMemcpyHostToDevice)); 
//   }

//   void SetupMatrix( float *K, float *R, float *Rot, float *Tsl)
//   {
//     // checkCUDA(__LINE__, cudaMemcpy(depth_gpu, depth_cpu, sizeof(float)*width*height, cudaMemcpyHostToDevice)); 

//     // memcpy(K_CPU, K, sizeof(float)*9);
//     // memcpy(R_CPU, R, sizeof(float)*9);
//     if (K_GPU == NULL) checkCUDA(__LINE__, cudaMalloc(&K_GPU, sizeof(float)*9));    
//     if (R_GPU == NULL) checkCUDA(__LINE__, cudaMalloc(&R_GPU, sizeof(float)*9));
//     if (Rot_GPU == NULL) checkCUDA(__LINE__, cudaMalloc(&Rot_GPU, sizeof(float)*4));
//     if (Tsl_GPU == NULL) checkCUDA(__LINE__, cudaMalloc(&Tsl_GPU, sizeof(float)*3));

//     checkCUDA(__LINE__, cudaMemcpy(K_GPU, (float*)K, sizeof(float)*9, cudaMemcpyHostToDevice));  
//     checkCUDA(__LINE__, cudaMemcpy(R_GPU, (float*)R, sizeof(float)*9, cudaMemcpyHostToDevice)); 
//     checkCUDA(__LINE__, cudaMemcpy(Rot_GPU, (float*)Rot, sizeof(float)*4, cudaMemcpyHostToDevice));  
//     checkCUDA(__LINE__, cudaMemcpy(Tsl_GPU, (float*)Tsl, sizeof(float)*3, cudaMemcpyHostToDevice)); 

//     // cout << "SetupMatrix:" << endl;
//     // cout << "K_GPU: " << K_GPU << endl;
//     // cout << "R_GPU: " << R_GPU << endl;
//     // cout << "Rot_GPU: " << Rot_GPU << endl;
//     // cout << "Tsl_GPU: " << Tsl_GPU << endl;
//   }

//   void ComputeXYZimage()
//   {
//     if (XYZimage_gpu!=NULL) checkCUDA(__LINE__, cudaFree(XYZimage_gpu));
//     checkCUDA(__LINE__, cudaMalloc(&XYZimage_gpu, sizeof(float)*width*height*3));
//     compute_xyzkernel<<<height,width>>>(XYZimage_gpu,depth_gpu,K_GPU,R_GPU);
//   }

//   void ComputeTSDF()
//   {
//     // checkCUDA(__LINE__, cudaMemcpy(depth_gpu, depth_cpu, sizeof(float)*width*height, cudaMemcpyHostToDevice)); 
//     // cout << "width: " << width << ", height: " << height << endl;
//     // cout << "memorysize: " << memorysize << endl;
//     // reallocateMemory();

//     // cout << depth_gpu << " " << depth_cpu << endl;
//     // cout << "Compute points: " << endl;
//     // checkCUDA(__LINE__, cudaMemcpy(depth_gpu, depth_cpu, sizeof(float)*width*height, cudaMemcpyHostToDevice)); 

//     // checkCUDA(__LINE__,cudaDeviceSynchronize());
//     compute_xyzkernel<<<height,width>>>(XYZimage_gpu,depth_gpu,K_GPU,R_GPU);
//     // checkCUDA(__LINE__,cudaDeviceSynchronize());

//     // float* p = new float[3*width*height];
//     // FILE* fp = fopen("debug2.bin", "wb");
//     // checkCUDA(__LINE__, cudaMemcpy(p, K_GPU, sizeof(float)*9, cudaMemcpyDeviceToHost));
//     // fwrite(p, sizeof(float), 9, fp);
//     // checkCUDA(__LINE__, cudaMemcpy(p, R_GPU, sizeof(float)*9, cudaMemcpyDeviceToHost));
//     // fwrite(p, sizeof(float), 9, fp);
//     // checkCUDA(__LINE__, cudaMemcpy(p, XYZimage_gpu, sizeof(float)*3*width*height, cudaMemcpyDeviceToHost));
//     // fwrite(&width, sizeof(unsigned int), 1, fp);
//     // fwrite(&height, sizeof(unsigned int), 1, fp);
//     // fwrite(p, sizeof(float), 3*width*height, fp);

//     // checkCUDA(__LINE__, cudaMemcpy(p, depth_gpu, sizeof(float)*width*height, cudaMemcpyDeviceToHost));
//     // fwrite(p, sizeof(float), width*height, fp);
    


//     // cout << "TSDF: " << endl;
//     int THREADS_NUM = 1024;
//     int BLOCK_NUM = int((grid_size + size_t(THREADS_NUM) - 1) / THREADS_NUM);
//     // compute_TSDFGPUbox_proj<<<BLOCK_NUM,THREADS_NUM>>>(tsdf_gpu, R_GPU, K_GPU, depth_gpu, XYZimage_gpu,
//                                       // scene_setting_gpu, width, height);

//     compute_TSDFGPU_grid_proj<<<BLOCK_NUM,THREADS_NUM>>>(tsdf_gpu, R_GPU, K_GPU, depth_gpu, XYZimage_gpu,
//                                       transform_grid_gpu, grid_size, truncated_value, width, height);

//     // compute_TSDFGPU_grid_proj_storageT<<<BLOCK_NUM,THREADS_NUM>>>(tsdf_gpu_storage, R_GPU, K_GPU, depth_gpu, XYZimage_gpu,
//     //                                   transform_grid_gpu, grid_size, truncated_value, width, height);
    
//     // delete[] p;
//     // p = new float[grid_size*3];
//     // checkCUDA(__LINE__, cudaMemcpy(p, tsdf_gpu, sizeof(float)*grid_size*3, cudaMemcpyDeviceToHost));
//     // fwrite(p, sizeof(float), grid_size*3, fp);
//     // fclose(fp);
//     // delete[] p;
//   }

//   void ComputeTSDF_storage()
//   {
//     // checkCUDA(__LINE__, cudaMemcpy(depth_gpu, depth_cpu, sizeof(float)*width*height, cudaMemcpyHostToDevice)); 
//     // cout << "width: " << width << ", height: " << height << endl;
//     // cout << "memorysize: " << memorysize << endl;
//     // reallocateMemory();

//     // cout << depth_gpu << " " << depth_cpu << endl;
//     // cout << "Compute points: " << endl;
//     // checkCUDA(__LINE__, cudaMemcpy(depth_gpu, depth_cpu, sizeof(float)*width*height, cudaMemcpyHostToDevice)); 

//     // checkCUDA(__LINE__,cudaDeviceSynchronize());
//     compute_xyzkernel<<<height,width>>>(XYZimage_gpu,depth_gpu,K_GPU,R_GPU);
//     // checkCUDA(__LINE__,cudaDeviceSynchronize());

//     // float* p = new float[3*width*height];
//     // FILE* fp = fopen("debug2.bin", "wb");
//     // checkCUDA(__LINE__, cudaMemcpy(p, K_GPU, sizeof(float)*9, cudaMemcpyDeviceToHost));
//     // fwrite(p, sizeof(float), 9, fp);
//     // checkCUDA(__LINE__, cudaMemcpy(p, R_GPU, sizeof(float)*9, cudaMemcpyDeviceToHost));
//     // fwrite(p, sizeof(float), 9, fp);
//     // checkCUDA(__LINE__, cudaMemcpy(p, XYZimage_gpu, sizeof(float)*3*width*height, cudaMemcpyDeviceToHost));
//     // fwrite(&width, sizeof(unsigned int), 1, fp);
//     // fwrite(&height, sizeof(unsigned int), 1, fp);
//     // fwrite(p, sizeof(float), 3*width*height, fp);

//     // checkCUDA(__LINE__, cudaMemcpy(p, depth_gpu, sizeof(float)*width*height, cudaMemcpyDeviceToHost));
//     // fwrite(p, sizeof(float), width*height, fp);
    


//     // cout << "TSDF: " << endl;
//     int THREADS_NUM = 1024;
//     int BLOCK_NUM = int((grid_size + size_t(THREADS_NUM) - 1) / THREADS_NUM);
//     // compute_TSDFGPUbox_proj<<<BLOCK_NUM,THREADS_NUM>>>(tsdf_gpu, R_GPU, K_GPU, depth_gpu, XYZimage_gpu,
//     //                                   scene_setting_gpu, width, height);

//     // compute_TSDFGPU_grid_proj<<<BLOCK_NUM,THREADS_NUM>>>(tsdf_gpu, R_GPU, K_GPU, depth_gpu, XYZimage_gpu,
//                                       // transform_grid_gpu, grid_size, truncated_value, width, height);


//     compute_TSDFGPU_grid_proj_storageT<<<BLOCK_NUM,THREADS_NUM>>>(tsdf_gpu_storage, R_GPU, K_GPU, depth_gpu, XYZimage_gpu, 
//                                                       transform_grid_gpu, grid_size, truncated_value, width, height);
    
//     // delete[] p;
//     // p = new float[grid_size*3];
//     // checkCUDA(__LINE__, cudaMemcpy(p, tsdf_gpu, sizeof(float)*grid_size*3, cudaMemcpyDeviceToHost));
//     // fwrite(p, sizeof(float), grid_size*3, fp);
//     // fclose(fp);
//     // delete[] p;
//   }

//   ~DepthImage()
//   {
//     cout << "Calling destructive function" << endl;

//     if (depth_cpu) delete[] depth_cpu;
//     // if (K_CPU) delete[] K_CPU;
//     // if (R_CPU) delete[] R_CPU;
    
//     if (K_GPU!=NULL) checkCUDA(__LINE__, cudaFree(K_GPU));
//     if (R_GPU!=NULL) checkCUDA(__LINE__, cudaFree(R_GPU));
//     if (tsdf_gpu!=NULL) checkCUDA(__LINE__, cudaFree(tsdf_gpu));
//     if (tsdf_gpu_storage!=NULL) checkCUDA(__LINE__, cudaFree(tsdf_gpu_storage));
//     if (scene_setting_gpu!=NULL) checkCUDA(__LINE__, cudaFree(scene_setting_gpu));
//     if (depth_gpu!=NULL) checkCUDA(__LINE__, cudaFree(depth_gpu));
//     if (XYZimage_gpu!=NULL) checkCUDA(__LINE__, cudaFree(XYZimage_gpu));

//     if (regular_grid_gpu!=NULL) checkCUDA(__LINE__, cudaFree(regular_grid_gpu));
//     if (transform_grid_gpu!=NULL) checkCUDA(__LINE__, cudaFree(transform_grid_gpu));
//   }
// };

// class SceneTemplate{
// public:
//   vector<Wall3D> walls;
//   vector<Box3D> objects;
//   vector<unsigned int> wall_valid;
//   vector<unsigned int> object_valid;
//   SceneTemplate()
//   {
//     walls.clear();
//     objects.clear();
//     wall_valid.clear();
//     object_valid.clear();
//   }
//   void transformTemplate(float* Rot, float* Tsl)
//   {
//     for (int i=0; i<walls.size(); i++)
//     {
//       walls[i].transformWall3D(Rot, Tsl);
//     }
//     for (int i=0; i<objects.size(); i++)
//     {
//       objects[i].transformBox3D(Rot, Tsl);
//     }
//   }

//   int numel_diff()
//   {
//     int dim = 0;
//     // cout << "walls: " << walls.size() << endl;
//     // cout << "objects: " << objects.size() << endl;
//     for (int i=0; i<walls.size(); i++)
//     {
//       if (walls[i].type == 1 || walls[i].type == 2)
//       {
//         dim += 1;
//       }
//       else if (walls[i].type == 0)
//       {
//         dim += 2;
//       }
//     }
//     for (int i=0; i<objects.size(); i++)
//     {
//       dim += 7;
//     }
//     return dim;
//   }

//   int numel_cls()
//   {
//     return walls.size() + objects.size();
//   }

//   void getRegressionValue( SceneTemplate &t, float* diff)
//   {
//     // diff = new float[numel_diff()];
//     int count = 0;
//     for (int i=0; i<walls.size(); i++)
//     {
//       if (walls[i].type == 1 || walls[i].type == 2)
//       {
//         diff[count] = walls[i].top_loc[0] - t.walls[i].top_loc[0];
//         count ++;
//       }
//       else if (walls[i].type == 0)
//       {
//         diff[count] = walls[i].top_loc[0] - t.walls[i].top_loc[0];
//         count++;
//         diff[count] = walls[i].top_loc[1] - t.walls[i].top_loc[1];
//         count++;
//       }
//     }
//     for (int i=0; i<objects.size(); i++)
//     {
//       diff[count] = objects[i].center[0] - t.objects[i].center[0];
//       count++;
//       diff[count] = objects[i].center[1] - t.objects[i].center[1];
//       count++;
//       diff[count] = objects[i].center[2] - t.objects[i].center[2];
//       count++;
//       diff[count] = objects[i].coeff[0] - t.objects[i].coeff[0];
//       count++;
//       diff[count] = objects[i].coeff[1] - t.objects[i].coeff[1];
//       count++;
//       diff[count] = objects[i].coeff[2] - t.objects[i].coeff[2];
//       count++;
//       diff[count] = objects[i].orientation[0] * t.objects[i].orientation[1] - objects[i].orientation[1] * t.objects[i].orientation[0];
//       count++;
//     }

//     // cout << "Regression Value: " << endl;
//     // for (int i = 0; i < count; ++i)
//     // {
//     //   cout << diff[i] << " ";
//     // }
//     // cout << endl;
//   }

//   void getRegressionValue_storage( SceneTemplate &t, StorageT* diff)
//   {
//     // diff = new float[numel_diff()];
//     int count = 0;
//     for (int i=0; i<walls.size(); i++)
//     {
//       if (walls[i].type == 1 || walls[i].type == 2)
//       {
//         diff[count] = CPUCompute2StorageT(ComputeT(walls[i].top_loc[0] - t.walls[i].top_loc[0]));
//         count ++;
//       }
//       else if (walls[i].type == 0)
//       {
//         diff[count] = CPUCompute2StorageT(ComputeT(walls[i].top_loc[0] - t.walls[i].top_loc[0]));
//         count++;
//         diff[count] = CPUCompute2StorageT(ComputeT(walls[i].top_loc[1] - t.walls[i].top_loc[1]));
//         count++;
//       }
//     }
//     for (int i=0; i<objects.size(); i++)
//     {
//       diff[count] = CPUCompute2StorageT(ComputeT(objects[i].center[0] - t.objects[i].center[0]));
//       count++;
//       diff[count] = CPUCompute2StorageT(ComputeT(objects[i].center[1] - t.objects[i].center[1]));
//       count++;
//       diff[count] = CPUCompute2StorageT(ComputeT(objects[i].center[2] - t.objects[i].center[2]));
//       count++;
//       diff[count] = CPUCompute2StorageT(ComputeT(objects[i].coeff[0] - t.objects[i].coeff[0]));
//       count++;
//       diff[count] = CPUCompute2StorageT(ComputeT(objects[i].coeff[1] - t.objects[i].coeff[1]));
//       count++;
//       diff[count] = CPUCompute2StorageT(ComputeT(objects[i].coeff[2] - t.objects[i].coeff[2]));
//       count++;
//       diff[count] = CPUCompute2StorageT(ComputeT(objects[i].orientation[0] * t.objects[i].orientation[1] - objects[i].orientation[1] * t.objects[i].orientation[0]));
//       count++;
//     }

//     // cout << "Regression Value: " << endl;
//     // for (int i = 0; i < count; ++i)
//     // {
//     //   cout << diff[i] << " ";
//     // }
//     // cout << endl;
//   }

//   void getRegressionBinary(float* diff)
//   {
//     // diff = new float[numel_diff()];
//     int count = 0;
//     for (int i=0; i<walls.size(); i++)
//     {
//       if (walls[i].type == 1 || walls[i].type == 2)
//       {
//         diff[count] = wall_valid[i];
//         count ++;
//       }
//       else if (walls[i].type == 0)
//       {
//         diff[count] = wall_valid[i];
//         count++;
//         diff[count] = wall_valid[i];
//         count++;
//       }
//     }
//     for (int i=0; i<objects.size(); i++)
//     {
//       diff[count] = object_valid[i];
//       count++;
//       diff[count] = object_valid[i];
//       count++;
//       diff[count] = object_valid[i];
//       count++;
//       diff[count] = object_valid[i];
//       count++;
//       diff[count] = object_valid[i];
//       count++;
//       diff[count] = object_valid[i];
//       count++;
//       diff[count] = object_valid[i];
//       count++;
//     }

//     // cout << "Regression Binary: " << endl;
//     // for (int i = 0; i < count; ++i)
//     // {
//     //   cout << diff[i] << " ";
//     // }
//     // cout << endl;
//   }

//   void getRegressionBinary_storage(StorageT* diff)
//   {
//     // diff = new float[numel_diff()];
//     int count = 0;
//     for (int i=0; i<walls.size(); i++)
//     {
//       if (walls[i].type == 1 || walls[i].type == 2)
//       {
//         diff[count] = CPUCompute2StorageT(ComputeT(wall_valid[i]));
//         count ++;
//       }
//       else if (walls[i].type == 0)
//       {
//         diff[count] = CPUCompute2StorageT(ComputeT(wall_valid[i]));
//         count++;
//         diff[count] = CPUCompute2StorageT(ComputeT(wall_valid[i]));
//         count++;
//       }
//     }
//     for (int i=0; i<objects.size(); i++)
//     {
//       diff[count] = CPUCompute2StorageT(ComputeT(object_valid[i]));
//       count++;
//       diff[count] = CPUCompute2StorageT(ComputeT(object_valid[i]));
//       count++;
//       diff[count] = CPUCompute2StorageT(ComputeT(object_valid[i]));
//       count++;
//       diff[count] = CPUCompute2StorageT(ComputeT(object_valid[i]));
//       count++;
//       diff[count] = CPUCompute2StorageT(ComputeT(object_valid[i]));
//       count++;
//       diff[count] = CPUCompute2StorageT(ComputeT(object_valid[i]));
//       count++;
//       diff[count] = CPUCompute2StorageT(ComputeT(object_valid[i]));
//       count++;
//     }

//     // cout << "Regression Binary: " << endl;
//     // for (int i = 0; i < count; ++i)
//     // {
//     //   cout << diff[i] << " ";
//     // }
//     // cout << endl;
//   }

//   void getClassificationValue(float* diff)
//   {
//     int count = 0;
//     for (int i=0; i<walls.size(); i++)
//     {
//       diff[count] = wall_valid[i];
//       count ++;
//     }
//     for (int i=0; i<objects.size(); i++)
//     {
//       diff[count] = object_valid[i];
//       count++;
//     } 

//     // cout << "Classification Value: " << endl;
//     // for (int i = 0; i < count; ++i)
//     // {
//     //   cout << diff[i] << " ";
//     // }
//     // cout << endl;
//   }

//   void getClassificationValue_storage(StorageT* diff)
//   {
//     int count = 0;
//     for (int i=0; i<walls.size(); i++)
//     {
//       diff[count] = CPUCompute2StorageT(ComputeT(wall_valid[i]));
//       count ++;
//     }
//     for (int i=0; i<objects.size(); i++)
//     {
//       diff[count] = CPUCompute2StorageT(ComputeT(object_valid[i]));
//       count++;
//     } 

//     // cout << "Classification Value: " << endl;
//     // for (int i = 0; i < count; ++i)
//     // {
//     //   cout << diff[i] << " ";
//     // }
//     // cout << endl;
//   }

//   void loadTemplate(FILE* &fp)
//   {
//     walls.clear();
//     objects.clear();
//     wall_valid.clear();
//     object_valid.clear();

//     unsigned int num_object_all;
//     fread(&num_object_all, 1, sizeof(unsigned int), fp);
//     for (int i = 0; i<num_object_all; ++i)
//     {
//       unsigned int is_object;
//       fread(&is_object, 1, sizeof(unsigned int), fp);
//       if (is_object==0)
//       {
//         walls.push_back(Wall3D());
//         walls[walls.size()-1].load(fp);
//       }
//       else if (is_object==1)
//       {
//         objects.push_back(Box3D());
//         objects[objects.size()-1].load(fp);
//       }
//     }

//     unsigned int *p = new unsigned int[walls.size()+objects.size()];
//     fread((void*)p, walls.size()+objects.size(), sizeof(unsigned int), fp);
//     int count = 0;
//     // cout << "wall_valid: ";
//     for (int i=0; i<walls.size(); i++,count++)
//     {
//       wall_valid.push_back(p[count]);
//     }
//     // for (int i=0; i<walls.size(); i++)
//     // {
//     //   cout << " " << wall_valid[i];
//     // }
//     // cout << endl;
//     // cout << "object_valid: ";
//     for (int i=0; i<objects.size(); i++,count++)
//     {
//       object_valid.push_back(p[count]);
//     }
//     // for (int i=0; i<objects.size(); i++)
//     // {
//     //   cout << " " << object_valid[i];
//     // }
//     // cout << endl;
//   }

// };


// class Scene3D{
// public:
//   // defined in .list file
//   string filename;
//   float K[9];
//   float R[9];
//   float Rot[4];
//   float Tsl[3];
//   unsigned int width;
//   unsigned int height;
//   SceneTemplate gnd;
  
//   void alignTemplate( SceneTemplate &t )
//   {


//   }

//   void loadDepthImage( DepthImage &depth)
//   {
//     depth.readImage(filename);
//   }

//   void loadCameraInfo(FILE* &fp)
//   {
//     fread((void*)Rot, sizeof(float), 4, fp);
//     fread((void*)Tsl, sizeof(float), 3, fp);
//     fread((void*)R,   sizeof(float), 9, fp);
//     fread((void*)K,   sizeof(float), 9, fp);
//     fread((void*)(&height), sizeof(unsigned int), 1, fp);
//     fread((void*)(&width),  sizeof(unsigned int), 1, fp);
    
//     // cout << "K: ";
//     // for (int i = 0; i<3; i++)
//     // {
//     //   cout << Tsl[i] << " ";
//     // }
//     // cout << endl;
//     // cout << height << " " << width << endl;
//   }

//   void loadTemplate(FILE* &fp)
//   {
//     gnd.loadTemplate(fp);
//   }

//   // void loadMetaData(FILE* &fp)
//   // {
//   //   unsigned int len = 0;
//   //   // cout << "len: " << len << endl;
//   //   // fread( (void*)(&len), sizeof(unsigned int), 1, fp); 
//   //   // fread( (void*)(filename.data()), sizeof(char), len, fp);
//   //   // cout << "len: " << len << ", filename: " << filename << endl;

//   //   fread( (void*)K, sizeof(float), 9, fp);
//   //   for (int i = 0; i<9; i++)
//   //   {
//   //     cout << K[i] << " ";
//   //   }
//   //   fread( (void*)R, sizeof(float), 9, fp);
//   //   for (int i = 0; i<9; i++)
//   //   {
//   //     cout << R[i] << " ";
//   //   }

//   //   fread( (void*)(&len), sizeof(unsigned int), 1, fp); 
//   //   cout << "len: " << len << endl;
//   //   for (int i=0; i<len; i++)
//   //   {
//   //     unsigned int entity_type;
//   //     fread( (void*)(&entity_type), sizeof(unsigned int), 1, fp);
//   //     cout << "entity_type: " <<  entity_type << endl;
//   //     if ( entity_type==0 )
//   //     {
//   //       walls.push_back(Wall3D());
//   //       walls[walls.size()-1].load(fp);
//   //     }
//   //     else if ( entity_type==1 )
//   //     {
//   //       objects.push_back(Box3D());
//   //       objects[objects.size()-1].load(fp);
//   //     }
//   //   }
//   //   fread((void*)(&height), sizeof(unsigned int), 1, fp);
//   //   fread((void*)(&width), sizeof(unsigned int), 1, fp);
//   // }

//   Scene3D() {
//     width = 0;
//     height = 0;
//   }

//   // Scene3D(FILE* &fp) {
//   //   loadMetaData(fp);
//   // }  
  
// };

// class DataManager{
// public:
//   void init( vector<int> &data_ids, vector<int> &labels, int num_label_, std::mt19937& rng_, bool balance_, bool shuffle_ )
//   {
//     if (balance_)
//     {
//       num_label = num_label_;

//       cls_counter.assign(num_label, 0);
//       iterator = 0;

//       cls_ids.assign(num_label, vector<int>(0,0));
//       for (int i = 0; i < data_ids.size(); ++i)
//       {
//         int id = data_ids[i];
//         cls_ids[labels[id]].push_back(i);
//       }

//       cls_order.assign(num_label, vector<size_t>(0,0));    
//     }
//     else
//     {
//       num_label = 1;
      
//       cls_counter.assign(num_label, 0);
//       iterator = 0;

//       cls_ids.assign(num_label, vector<int>(0,0));
//       cls_ids[0].resize( data_ids.size());
//       for (int i = 0; i < data_ids.size(); ++i)
//       {
//         cls_ids[0][i] = i;
//       }

//       cls_order.assign(num_label, vector<size_t>(0,0));
//     }

//     rng = rng_;
//     shuffle = shuffle_;
//     shuffle_all();
//   }

//   void shuffle_all( )
//   {
//     for (int i = 0; i < num_label; ++i)
//     {
//       shuffle_cls(i);
//     }
//   }

//   void shuffle_cls( int cls )
//   {
//     if (shuffle)
//       cls_order[cls] = randperm( cls_ids[cls].size(), rng);
//     else {
//       cls_order[cls].resize( cls_ids[cls].size() );
//       for (int i = 0; i < cls_ids[cls].size(); ++i ) cls_order[cls][i] = i;
//     }
//   }

//   int next_id( int &epoch)
//   {
//     // cout << "iterator: " << iterator << "; cls_counter: " << cls_counter[iterator] << "; cls_order: " << cls_order[iterator][cls_counter[iterator]] <<endl;

//     int id1 = cls_order[iterator][cls_counter[iterator]];
//     int next_id = cls_ids[iterator][id1];

//     cls_counter[iterator] ++;
//     if ( cls_counter[iterator] >= cls_order[iterator].size() )
//     {
//       cls_counter[iterator] = 0;
//       shuffle_cls(iterator);
//       if (num_label==1) epoch++;
//     }

//     iterator ++;
//     if ( iterator >= num_label )
//     {
//       iterator = 0;
//     }

//     return next_id;
//   }

//   vector<vector<int>> cls_ids;
//   vector<vector<size_t>> cls_order;
//   vector<int> cls_counter;
//   int num_label;
//   int iterator;
//   bool shuffle;
//   std::mt19937 rng;
// };


///////////// new layer
// Yinda: a layer read depth image, and compute tsdf online
class SceneHolisticOnlineTSDFLayer : public DataLayer {
public:
  int epoch_prefetch;

  string data_root;
  vector<string> data_list_name;
  vector<int> data_list_id;
  vector<int> label_list;

  string template_file;
  string file_list;
  string file_label;
  string file_norm;
  string ground_truth_file;
  string camera_info_file;
  string scene_weight;
  string object_weight;
  
  vector<int> grid_size;
  vector<float> spatial_range;
  // int GPU;
  
  int batch_size;
  future<void> lock;

  FILE* dataFILE;
  vector<size_t> ordering;

  vector<SceneTemplate> scene_templates;
  DepthImage depth_image;
  vector<Scene3D> ground_truth;

  unsigned int num_scene_template;
  unsigned int num_scene_3d;
  vector<int> regression_len;
  vector<int> class_len;

  vector<int> data_dims;
  vector<int> bbox_regression_dims;
  vector<int> bbox_classification_dims;
  vector<int> noise_value_dims;
  vector<int> scene_classification_dims;
  vector<int> bbox_classification_c_dims;

  vector<float> mean_value;
  vector<float> std_value;

  vector<float> scene_cls_weight;
  vector<float> object_cls_weight;

  vector<float> noise_scale;

  StorageT* data_gpu;
  StorageT* bbox_reg_val_gpu;
  StorageT* bbox_reg_val_cpu;
  StorageT* bbox_reg_bin_gpu;
  StorageT* bbox_reg_bin_cpu;
  StorageT* bbox_cls_val_gpu;
  StorageT* bbox_cls_val_cpu;
  StorageT* bbox_cls_bin_gpu;
  StorageT* bbox_cls_bin_cpu;
  StorageT* scene_type_cpu;
  StorageT* scene_type_gpu;
  StorageT* noise_value_gpu;
  StorageT* noise_value_cpu;


  int numel_batch_data;
  int numel_single_data;
  int numel_batch_bbox_regression;
  int numel_single_bbox_regression;
  int numel_batch_bbox_classification;
  int numel_single_bbox_classification;
  int numel_batch_noise;
  int numel_single_noise;
  int numel_batch_scene_classification;
  int numel_single_scene_classification;
  int numel_batch_bbox_c_classification;
  int numel_single_bbox_c_classification;
  bool shuffle_data;
  bool balance_data;

  float NoiseSeed[20][3] = 
  {
    {-0.665851451573081,  -0.0157468111080493,  -0.157243160564860},
    {-0.824478864937384,  0.115659371451520,  -0.0834037258450930},
    {-0.330708955801585,  -0.465952044794167, 0.136139998605281},
    {0.191343238138851, -0.376017434940846, -0.115500279151007},
    {0.0108408652573738,  -0.949422202003150, -0.457256282914468},
    {0.0689424226069920,  -0.677855084076204, 0.580851972372071},
    {-0.196000934777747,  -0.0502921060543901,  0.323391784720610},
    {0.320991822968146, 0.150932121121586,  -0.516590276391963},
    {0.450743710838035, -0.932682261394910, 0.477017584302802},
    {0.734893088153549, -1.35415154922103,  0.748866514904553},
    {-0.239300125954518,  0.903930982794089,  -0.0836206703793616},
    {-0.734354195823346,  -0.195436044323641, -0.210215612747387},
    {0.459610707649813, 0.593948491672545,  0.119207581100844},
    {-0.657577127353686,  -0.0288784370796805,  0.424992445921232},
    {-0.614372181171440,  -0.756443705462201, 0.155802107013795},
    {-0.426934274785244,  -0.785849021870207, 0.564021496234610},
    {-0.235712667316220,  0.0617955788477067, -0.379863700574596},
    {-0.565936673922783,  -0.345970297581263, -0.0412110843219868},
    {-0.203472885863948,  1.21206236489750, 0.493722303071250},
    {0.768644504963014, 0.290156921253273,  -0.494798195205457}
  };

  DataManager id_manager;

  // int map_weight_valid[8];

  int numofitems(){
    return data_list_name.size();
  };

  int numofitemsTruncated(){
    return batch_size * floor(double(numofitems())/double(batch_size));
  };

  void init(){

    // StorageT test[10];
    // memset(test, 0, 10*sizeofStorageT);
    // for (int i = 0; i < 10; ++i)
    // {
    //   std::cout << CPUStorage2ComputeT(test[i]) << std::endl;
    // }
    // cerr<<"Debug Stop"<<endl;
    // FatalError(__LINE__);


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
    // cout << "[0]: " << data_list_name[0] << endl;
    // cout << "[end]: " << data_list_name[num_data-1] << endl;
    cout << data_list_name.size() << "data load!" <<  endl;
    shuffle();
    counter = 0;

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

    ////////////////////// read normalization
    dataFILE = fopen( file_norm.c_str(), "r");
    fread( (void *)(&iBuffer), sizeof(unsigned int), 1, dataFILE);
    mean_value.clear();
    std_value.clear();
    float fBuffer;
    for (int i = 0; i < iBuffer; ++i)
    {
      fread( (void *)(&fBuffer), sizeof(float), 1, dataFILE);
      mean_value.push_back(fBuffer);
    }
    for (int i = 0; i < iBuffer; ++i)
    {
      fread( (void *)(&fBuffer), sizeof(float), 1, dataFILE);
      std_value.push_back(fBuffer);
    }
    fclose(dataFILE);
    // cout << "mean: " << veciPrintString(mean_value) << endl;
    // cout << "std: " << veciPrintString(std_value) << endl;

    ////////////////////// read scene template
    cout << "Read template..." << endl;
    dataFILE = fopen(template_file.c_str(), "rb");
    if (dataFILE==NULL){
      cerr<<"Fail to open the data file"<<endl;
      FatalError(__LINE__);
    }
    fread( (void*)(&num_scene_template), sizeof(unsigned int), 1, dataFILE);
    cout << "num_scene_template: " << num_scene_template << endl;
    scene_templates.assign(num_scene_template, SceneTemplate());
    for (int sid=0; sid<num_scene_template; sid++)
    {
      scene_templates[sid].loadTemplate(dataFILE);
    }
    fclose(dataFILE);

    regression_len.assign( num_scene_template+1, 0);
    class_len.assign( num_scene_template+1, 0);
    // cout << "see here:" << scene_templates[0].walls.size() << " " << scene_templates[0].objects.size() << endl;
    for (int sid=0; sid<num_scene_template; sid++)
    {
      regression_len[sid+1] = regression_len[sid] + scene_templates[sid].numel_diff();
      class_len[sid+1] = class_len[sid] + scene_templates[sid].numel_cls();
    }
    // cout << "regression_len: " << regression_len << endl;
    // cout << "class_len: " << class_len << endl;
    // cout << "regression_len: " << veciPrintString(regression_len) << endl;
    // cout << "class_len: " << veciPrintString(class_len) << endl;

    ////////////////////// read weights
    dataFILE = fopen(scene_weight.c_str(), "rb");
    scene_cls_weight.clear();
    for (int sid = 0; sid<num_scene_template; sid++)
    {
      fread( (void *)(&fBuffer), sizeof(float), 1, dataFILE);
      scene_cls_weight.push_back(fBuffer);
    }
    fclose(dataFILE);

    dataFILE = fopen(object_weight.c_str(), "rb");
    object_cls_weight.clear();
    for (int sid = 0; sid<2*class_len.back(); sid++)
    {
      fread( (void *)(&fBuffer), sizeof(float), 1, dataFILE);
      object_cls_weight.push_back(fBuffer);
    }
    fclose(dataFILE);
    cout << "object_cls_weight: " << vecPrintString(object_cls_weight) << endl;

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
    for (int sid=0; sid<num_scene_3d; sid++)
    {
      ground_truth[sid].loadCameraInfo(dataFILE);
    }
    fclose(dataFILE);

    ////////////////////// Prepare for TSDF computing
    // depth_images.assign(1, DepthImage());
    float temp[10];
    temp[0] = spatial_range[0]; temp[1] = spatial_range[1];
    temp[2] = spatial_range[2]; temp[3] = spatial_range[3];
    temp[4] = spatial_range[4]; temp[5] = spatial_range[5];
    temp[6] = grid_size[3]; temp[7] = grid_size[2]; temp[8] = grid_size[1];
    temp[9] = 3.0 * (spatial_range[1] - spatial_range[0]) / grid_size[3];
    cout << "temp[9] = " << temp[9] << endl;
    depth_image.SceneSetting(temp);

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

    bbox_regression_dims.clear();
    bbox_regression_dims.push_back(batch_size);
    bbox_regression_dims.push_back(regression_len.back());
    bbox_regression_dims.push_back(1);
    bbox_regression_dims.push_back(1);
    bbox_regression_dims.push_back(1);
    numel_batch_bbox_regression = numel(bbox_regression_dims);
    numel_single_bbox_regression = sizeofitem(bbox_regression_dims);
    cout << "bbox_regression_dims: " << vecPrintString(bbox_regression_dims) << endl;
    cout << "numel_batch_bbox_regression: " << numel_batch_bbox_regression << endl;
    cout << "numel_single_bbox_regression: " << numel_single_bbox_regression << endl;

    bbox_classification_dims.clear();
    bbox_classification_dims.push_back(batch_size);
    bbox_classification_dims.push_back(1);
    bbox_classification_dims.push_back(class_len.back());
    bbox_classification_dims.push_back(1);
    bbox_classification_dims.push_back(1);
    numel_batch_bbox_classification = numel(bbox_classification_dims);
    numel_single_bbox_classification = sizeofitem(bbox_classification_dims);
    cout << "bbox_classification_dims: " << vecPrintString(bbox_classification_dims) << endl;
    cout << "numel_batch_bbox_classification: " << numel_batch_bbox_classification << endl;
    cout << "numel_single_bbox_classification: " << numel_single_bbox_classification << endl;

    bbox_classification_c_dims.clear();
    bbox_classification_c_dims.push_back(batch_size);
    bbox_classification_c_dims.push_back(2);
    bbox_classification_c_dims.push_back(class_len.back());
    bbox_classification_c_dims.push_back(1);
    bbox_classification_c_dims.push_back(1);
    numel_batch_bbox_c_classification = numel(bbox_classification_c_dims);
    numel_single_bbox_c_classification = sizeofitem(bbox_classification_c_dims);
    cout << "bbox_classification_c_dims: " << vecPrintString(bbox_classification_c_dims) << endl;
    cout << "numel_batch_bbox_c_classification: " << numel_batch_bbox_c_classification << endl;
    cout << "numel_single_bbox_c_classification: " << numel_single_bbox_c_classification << endl;

    scene_classification_dims.clear();
    scene_classification_dims.push_back(batch_size);
    scene_classification_dims.push_back(1);
    scene_classification_dims.push_back(1);
    scene_classification_dims.push_back(1);
    scene_classification_dims.push_back(1);
    numel_batch_scene_classification = numel(scene_classification_dims);
    numel_single_scene_classification = sizeofitem(scene_classification_dims);
    cout << "scene_classification_dims: " << vecPrintString(scene_classification_dims) << endl;
    cout << "numel_batch_scene_classification: " << numel_batch_scene_classification << endl;
    cout << "numel_single_scene_classification: " << numel_single_scene_classification << endl;

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
    bbox_reg_val_cpu = new StorageT[numel_batch_bbox_regression];
    bbox_reg_bin_cpu = new StorageT[numel_batch_bbox_regression];
    bbox_cls_val_cpu = new StorageT[numel_batch_bbox_classification];
    bbox_cls_bin_cpu = new StorageT[numel_batch_bbox_c_classification];
    scene_type_cpu = new StorageT[numel_batch_scene_classification];
    noise_value_cpu = new StorageT[numel_batch_noise];

    checkCUDA(__LINE__, cudaMalloc(&data_gpu, numel_batch_data * sizeofStorageT) );
    checkCUDA(__LINE__, cudaMalloc(&bbox_reg_val_gpu, numel_batch_bbox_regression * sizeofStorageT) );
    checkCUDA(__LINE__, cudaMalloc(&bbox_reg_bin_gpu, numel_batch_bbox_regression * sizeofStorageT) );
    checkCUDA(__LINE__, cudaMalloc(&bbox_cls_val_gpu, numel_batch_bbox_classification * sizeofStorageT) );
    checkCUDA(__LINE__, cudaMalloc(&bbox_cls_bin_gpu, numel_batch_bbox_c_classification * sizeofStorageT) );
    checkCUDA(__LINE__, cudaMalloc(&scene_type_gpu, numel_batch_scene_classification * sizeofStorageT) );
    checkCUDA(__LINE__, cudaMalloc(&noise_value_gpu, numel_batch_noise * sizeofStorageT) );

    ///////////// initialize 
    // NoiseSeed = 
    // {
    //  {-0.665851451573081,  -0.0157468111080493,  -0.157243160564860},
    //  {-0.824478864937384,  0.115659371451520,  -0.0834037258450930},
    //  {-0.330708955801585,  -0.465952044794167, 0.136139998605281},
    //  {0.191343238138851, -0.376017434940846, -0.115500279151007},
    //  {0.0108408652573738,  -0.949422202003150, -0.457256282914468},
    //  {0.0689424226069920,  -0.677855084076204, 0.580851972372071},
    //  {-0.196000934777747,  -0.0502921060543901,  0.323391784720610},
    //  {0.320991822968146, 0.150932121121586,  -0.516590276391963},
    //  {0.450743710838035, -0.932682261394910, 0.477017584302802},
    //  {0.734893088153549, -1.35415154922103,  0.748866514904553},
    //  {-0.239300125954518,  0.903930982794089,  -0.0836206703793616},
    //  {-0.734354195823346,  -0.195436044323641, -0.210215612747387},
    //  {0.459610707649813, 0.593948491672545,  0.119207581100844},
    //  {-0.657577127353686,  -0.0288784370796805,  0.424992445921232},
    //  {-0.614372181171440,  -0.756443705462201, 0.155802107013795},
    //  {-0.426934274785244,  -0.785849021870207, 0.564021496234610},
    //  {-0.235712667316220,  0.0617955788477067, -0.379863700574596},
    //  {-0.565936673922783,  -0.345970297581263, -0.0412110843219868},
    //  {-0.203472885863948,  1.21206236489750, 0.493722303071250},
    //  {0.768644504963014, 0.290156921253273,  -0.494798195205457}
    // };


    ///////////// compute 
    // checkCUDA(__LINE__,cudaDeviceSynchronize());
    // cout << "In init()" << endl;
    // string data_file = data_root + "0001_depth.bin";
    // depth_image.readImage(data_file);
    // cout << "SetupMatrix" << endl;
    // depth_image.SetupMatrix(ground_truth[0].K, ground_truth[0].R, ground_truth[0].Rot, ground_truth[0].Tsl);
    // cout << "Transform Grid point" << endl;
    // depth_image.transformGridPoints();
    // // cout << "ComputeTSDF" << endl;
    // // depth_image.ComputeTSDF();
    // cout << "init() finish" << endl;
    // checkCUDA(__LINE__,cudaDeviceSynchronize());
    // cout << "In init() 2" << endl;
    // data_file = data_root + "0002_depth.bin";
    // depth_image.readImage(data_file);
    // cout << "SetupMatrix" << endl;
    // depth_image.SetupMatrix(ground_truth[1].K, ground_truth[1].R, ground_truth[1].Rot, ground_truth[1].Tsl);
    // cout << "Transform Grid point" << endl;
    // depth_image.transformGridPoints();
    // // cout << "ComputeTSDF" << endl;
    // // depth_image.ComputeTSDF();
    // cout << "init() finish 2" << endl;
    // checkCUDA(__LINE__,cudaDeviceSynchronize());

    cout << "noise_scale: " << noise_scale[0] << " " << noise_scale[1] << " " << noise_scale[2] << endl;

    id_manager.init( data_list_id, label_list, 8, rng , balance_data, shuffle_data );

    // shuffle();
    // cout << "1st id: " << ordering[0] << endl;

    // if (phase!=Testing){
    //  shuffle();
    // }else{
    //  ordering.resize(data_list_name.size());
    //  for (int i = 0; i < data_list_name.size(); ++i ) ordering[i]=i;
    // }

  };

  SceneHolisticOnlineTSDFLayer(string name_, Phase phase_): DataLayer(name_){
    phase = phase_;
    init();
  };

  SceneHolisticOnlineTSDFLayer(JSON* json){
    SetValue(json, name,    "Whatever")
    SetValue(json, phase,   Training)
    SetOrDie(json, template_file )
    SetOrDie(json, ground_truth_file )
    SetOrDie(json, camera_info_file )
    SetOrDie(json, file_list )
    SetOrDie(json, file_label )
    SetOrDie(json, file_norm)
    // SetOrDie(json, GPU )
    SetOrDie(json, data_root )
    SetOrDie(json, grid_size )
    SetOrDie(json, spatial_range )
    SetValue(json, batch_size,  64)
    SetValue(json, shuffle_data, true);
    SetValue(json, balance_data, true);

    SetOrDie(json, scene_weight )
    SetOrDie(json, object_weight )
    SetValue(json, noise_scale, vector<float>(3,1.5))

    cout << "template_file:　" << template_file << endl;
    cout << "ground_truth_file:　" << ground_truth_file << endl;
    cout << "file_list:　" << file_list << endl;
    cout << "file_label: " << file_label << endl;
    cout << "camera_info_file: " << camera_info_file << endl;
    cout << "data_root: " << data_root << endl;
    cout << "file_norm: " << file_norm << endl;

    cout << "grid_size: " << vecPrintString(grid_size) << endl;
    cout << "spatial_range: " << vecPrintString(spatial_range) << endl;
    cout << "batch_size: " << batch_size << endl;
    cout << "noise_scale: " << vecPrintString(noise_scale) << endl;
    init();
  };

  ~SceneHolisticOnlineTSDFLayer(){
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

    // checkCUDA(__LINE__,cudaSetDevice(GPU));
    checkCUDA(__LINE__,cudaSetDevice(GPU));
    // tic(); cout<<"read disk  ";
    size_t read_count;
    
    memset( (void*)bbox_reg_val_cpu, 0, sizeofStorageT * numel_batch_bbox_regression);
    memset( (void*)bbox_reg_bin_cpu, 0, sizeofStorageT * numel_batch_bbox_regression);
    memset( (void*)bbox_cls_val_cpu, 0, sizeofStorageT * numel_batch_bbox_classification);
    memset( (void*)bbox_cls_bin_cpu, 0, sizeofStorageT * numel_batch_bbox_c_classification);
    memset( (void*)scene_type_cpu, 0, sizeofStorageT * numel_batch_scene_classification);
    memset( (void*)noise_value_cpu, 0, sizeofStorageT * numel_batch_noise);

    for (size_t i = 0; i < batch_size; ++i)
    {
      int local_id = id_manager.next_id(epoch_prefetch);
      // cout << "local_id: " << local_id << endl;

      string data_name = data_list_name[local_id];
      string data_file = data_root + data_name + "_depth.bin";    
      int data_id = data_list_id[local_id];

      // cout << "data_id: " << data_id << "; label: " << label_list[data_id] << endl;

      // string data_name = data_list_name[ordering[counter]];
      // string data_file = data_root + data_name + "_depth.bin";     
      // int data_id = data_list_id[ordering[counter]];

      // cout << "data_file: " << data_file << endl;
      // cout << "data_id: " << data_id << endl;
      // cout << "scene_type: " << label_list[data_id] << endl;
      // cout << "R: " << endl;
      // for (int t = 0; t<9; t++)
      //  cout << ground_truth[data_id].R[t] << " ";
      // cout << endl;
      // cout << "K: " << endl;
      // for (int t = 0; t<9; t++)
      //  cout << ground_truth[data_id].K[t] << " ";
      // cout << endl;
      // cout << "Rot: " << endl;
      // for (int t = 0; t<4; t++)
      //  cout << ground_truth[data_id].Rot[t] << " ";
      // cout << endl;
      // cout << "Tsl: " << endl;
      // for (int t = 0; t<3; t++)
      //  cout << ground_truth[data_id].Tsl[t] << " ";
      // cout << endl;

      // data_file = data_root + "001129_depth.bin";
      // data_id = 1128;
      // float* bbox_reg_val_target = bbox_reg_val_cpu + i * numel_single_bbox_regression + regression_len[label_id];
      // float* bbox_reg_bin_target = bbox_reg_bin_cpu + i * numel_single_bbox_regression + regression_len[label_id];
      // float* bbox_cls_val_target = bbox_cls_val_cpu + i * numel_single_bbox_classification + class_len[label_id];
      // float* bbox_cls_bin_target = bbox_cls_bin_cpu + i * numel_single_bbox_classification + class_len[label_id];
      // float* scene_type_target = scene_type_cpu + i * numel_single_scene_classification;
      // float* noise_value_target = noise_value_cpu + i * numel_single_noise;

      ///////////// read data

      // string ttt = "../Deep3DScene/ProtoType_V2/image_V1/0156_depth.bin";
      // depth_images[0].readImage(ttt);
      
      // string ttt = data_file;
      // depth_images[0].readImage(ttt);
      // cout << "before depth" << endl;
      // cout << "read from: " << data_file << endl;
      depth_image.readImage(data_file);
      // cout << "after depth" << endl;

      ///////////// add noise
      float noise_dif[3];
      // int noise_id = rand()%20;
      // int noise_id = 0;
      // noise_dif[0] = NoiseSeed[noise_id][0];
      // noise_dif[1] = NoiseSeed[noise_id][1];
      // noise_dif[2] = NoiseSeed[noise_id][2];
      noise_dif[0] = 2.0*(float)(rand()%100)/100*noise_scale[0] - noise_scale[0];
      noise_dif[1] = 2.0*(float)(rand()%100)/100*noise_scale[1] - noise_scale[1];
      noise_dif[2] = 2.0*(float)(rand()%100)/100*noise_scale[2] - noise_scale[2];

      // float rotate_dif[9] = {1,0,0,1};

      // cout << "noise_dif: " << noise_dif[0] << " " << noise_dif[1] << " " << noise_dif[2] << endl;

      float noise_tsl[3];
      noise_tsl[0] = ground_truth[data_id].Tsl[0] + noise_dif[0];
      noise_tsl[1] = ground_truth[data_id].Tsl[1] + noise_dif[1];
      noise_tsl[2] = ground_truth[data_id].Tsl[2] + noise_dif[2];
      // cout << "noise_tsl: " << noise_tsl[0] << " " << noise_tsl[1] << " " << noise_tsl[2] << endl;

      ///////////// compute TSDF
      // cout << "Enter prefetch: " << endl;
      // cout << "SetupMatrix" << endl;
      // depth_image.SetupMatrix(ground_truth[data_id].K, ground_truth[data_id].R, ground_truth[data_id].Rot, noise_tsl);
      depth_image.SetupMatrix(ground_truth[data_id].K, ground_truth[data_id].R, ground_truth[data_id].Rot, noise_tsl);
      // cout << "Transform Grid point" << endl;
      depth_image.transformGridPoints();
      // cout << "ComputeTSDF" << endl;
      // depth_image.ComputeTSDF();
      depth_image.ComputeTSDF_storage();
      // cout <<"prefetch end" <<  endl;

      checkCUDA(__LINE__, cudaMemcpy( data_gpu + i*numel_single_data,  depth_image.tsdf_gpu_storage,  numel_single_data * sizeofStorageT, cudaMemcpyDeviceToDevice) );
      // checkCUDA(__LINE__, cudaMemcpy( data_gpu + i*numel_single_data,  GPUCompute2StorageT((ComputeT*)depth_image.tsdf_gpu),  numel_single_data * sizeofStorageT, cudaMemcpyDeviceToDevice) );
      // GPUCompute2StorageT

      ///////////// calculate bbox pred value
      SceneTemplate transform_gnd = ground_truth[data_id].gnd;
      transform_gnd.transformTemplate(ground_truth[data_id].Rot, noise_tsl);
      // transform_gnd.transformTemplate( rotate_dif, noise_dif);
      // transform_gnd.getRegressionValue( ground_truth[data_id].gnd);

      int label_id = label_list[data_id];

      StorageT* bbox_reg_val_target = bbox_reg_val_cpu + i * numel_single_bbox_regression;
      StorageT* bbox_reg_bin_target = bbox_reg_bin_cpu + i * numel_single_bbox_regression;
      StorageT* bbox_cls_val_target = bbox_cls_val_cpu + i * numel_single_bbox_classification;
      StorageT* bbox_cls_bin_target = bbox_cls_bin_cpu + i * numel_single_bbox_c_classification;
      StorageT* scene_type_target = scene_type_cpu + i * numel_single_scene_classification;
      StorageT* noise_value_target = noise_value_cpu + i * numel_single_noise;
      
      // memcpy( noise_value_target, noise_dif, sizeofVtype*3);
      for (int j = 0; j<3; j++)
      {
        noise_value_target[j] = CPUCompute2StorageT( ComputeT(noise_dif[j]) );
      }

      // *scene_type_target = label_id;
      *scene_type_target = CPUCompute2StorageT( ComputeT(label_id) );

      // for (int j = class_len[label_id]; j<class_len[label_id+1]; j++)
      // {
      //   *(bbox_cls_bin_target+j) = 1;
      // }
      for (int j = class_len[label_id]; j < class_len[label_id+1]; j++)
      {
        bbox_cls_bin_target[j] = CPUCompute2StorageT( ComputeT(object_cls_weight[j]) );
        bbox_cls_bin_target[j+class_len.back()] = CPUCompute2StorageT( ComputeT(object_cls_weight[j+class_len.back()]) );
      }


      transform_gnd.getClassificationValue_storage( bbox_cls_val_target + class_len[label_id] );
      transform_gnd.getRegressionBinary_storage( bbox_reg_bin_target + regression_len[label_id] );

      // if (phase == Training)
      // {
      //  cout << "bbox_reg_val: ";
      //  for (int k = 0; k < numel_single_bbox_regression; k++)
      //  {
      //    cout << *(bbox_reg_val_target+k) << " ";
      //  }
      //  cout << endl;
      // }

      scene_templates[label_id].getRegressionValue_storage( transform_gnd, bbox_reg_val_target + regression_len[label_id] );

      // if (phase == Training)
      // {
      //  cout << "scene_template: " << scene_templates[label_id].walls[0].top_loc[0] << endl;
      //  cout << "trainsform: " << transform_gnd.walls[0].top_loc[0] << endl;
      //  cout << "Rot: ";
      //  for (int k = 0; k<4; k++) cout << ground_truth[data_id].Rot[k] << " ";
      //  cout << endl;
      //  for (int k = 0; k<3; k++) cout << ground_truth[data_id].Tsl[k] << " ";
      //  cout << endl;

      //  cout << "bbox_reg_val: ";
      //  for (int k = 0; k < numel_single_bbox_regression; k++)
      //  {
      //    cout << *(bbox_reg_val_target+k) << " ";
      //  }
      //  cout << endl;
      // }

      for (int j = regression_len[label_id]; j < regression_len[label_id+1]; ++j)
      {
        float invalue = CPUStorage2ComputeT(bbox_reg_val_target[j]);
        float outvalue = (invalue - mean_value[j]) / std_value[j] * 10;
        *(bbox_reg_val_target+j) = CPUCompute2StorageT( ComputeT(outvalue) );
      }
      // for (int j = class_len[label_id]; j < class_len[label_id+1]; j++)
      // {
      //   float invalue = CPUStorage2ComputeT(bbox_cls_val_target[j]);
      //   float outvalue = invalue * 5;
      //   *(bbox_cls_val_target+j) = CPUCompute2StorageT( ComputeT(outvalue) );
      // }

      // transform_gnd.getRegrssionValue( ground_truth[data_id].gnd, bbox_reg_val_target + regression_len[label_id] );
      // counter++;
      // if ( counter >= ordering.size()){
      //   // if (phase!=Testing) shuffle();
      //   shuffle();
      //   counter = 0;
      //   ++epoch_prefetch;
      // } 


      //////////////// debug
      // if (phase == Training)
      // {
      //  cerr<<"Compute first data, and Stop"<<endl;
      //  FatalError(__LINE__);
      // }

      // if (phase == Training)
      // {
      //  cout << "data_file: " << data_file << endl;

      //  cout << "scene_classification: ";
      //  for (int k = 0; k < numel_single_scene_classification; k++)
      //  {
      //    cout << *(scene_type_target+k) << " ";
      //  }
      //  cout << endl;

      //  cout << "bbox_reg_val: ";
      //  for (int k = 0; k < numel_single_bbox_regression; k++)
      //  {
      //    cout << *(bbox_reg_val_target+k) << " ";
      //  }
      //  cout << endl;

      //  cout << "bbox_reg_bin: ";
      //  for (int k = 0; k < numel_single_bbox_regression; k++)
      //  {
      //    cout << *(bbox_reg_bin_target+k) << " ";
      //  }
      //  cout << endl;

      //  cout << "bbox_cls_bin: ";
      //  for (int k = 0; k < numel_single_bbox_classification; k++)
      //  {
      //    cout << *(bbox_cls_bin_target+k) << " ";
      //  }
      //  cout << endl;

      //  cout << "R: ";
      //  for (int k = 0; k < 9; k++)
      //  {
      //    cout << ground_truth[data_id].R[k] << " ";
      //  }
      //  cout << endl;

      // //   cout << "save" << endl;
      // //   FILE* fp = fopen("debug_data.bin", "wb");
      // //   fwrite( &data_id, sizeof(int), 1, fp);
      // //   fwrite( noise_dif, sizeof(float), 3, fp);

      // //   float* p = new float[depth_image.grid_size*3];

      // //   checkCUDA(__LINE__, cudaMemcpy( p,  depth_image.transform_grid_gpu,  depth_image.grid_size * 3 * sizeofVtype, cudaMemcpyDeviceToHost) );
      // //   fwrite( p, sizeof(float), depth_image.grid_size * 3, fp);

      // //   checkCUDA(__LINE__, cudaMemcpy( p,  depth_image.tsdf_gpu,  depth_image.grid_size * 3 * sizeofVtype, cudaMemcpyDeviceToHost) );
      // //   fwrite( p, sizeof(float), depth_image.grid_size * 3, fp);

      // //   delete[] p;
      // //   p = new float[depth_image.width * depth_image.height * 3];
      // //   checkCUDA(__LINE__, cudaMemcpy( p,  depth_image.XYZimage_gpu,  depth_image.width * depth_image.height * 3 * sizeofVtype, cudaMemcpyDeviceToHost) );
      // //   fwrite( p, sizeof(float), depth_image.width * depth_image.height * 3, fp);

      // //   fclose(fp);
      //  cerr<<"Compute first data, and Stop"<<endl;
      //  FatalError(__LINE__);
  
      // }

      

    }
    
    /////////////////// ship to gpu

    // checkCUDA(__LINE__, cudaMemcpy( data_gpu,  depth_image.tsdf_gpu,  numel_batch_data * sizeofVtype, cudaMemcpyDeviceToDevice) );

    checkCUDA(__LINE__, cudaMemcpy( bbox_reg_val_gpu, bbox_reg_val_cpu, numel_batch_bbox_regression * sizeofStorageT, cudaMemcpyHostToDevice) );
    checkCUDA(__LINE__, cudaMemcpy( bbox_reg_bin_gpu, bbox_reg_bin_cpu, numel_batch_bbox_regression * sizeofStorageT, cudaMemcpyHostToDevice) );

    checkCUDA(__LINE__, cudaMemcpy( bbox_cls_val_gpu, bbox_cls_val_cpu, numel_batch_bbox_classification * sizeofStorageT, cudaMemcpyHostToDevice) );
    checkCUDA(__LINE__, cudaMemcpy( bbox_cls_bin_gpu, bbox_cls_bin_cpu, numel_batch_bbox_c_classification * sizeofStorageT, cudaMemcpyHostToDevice) );

    checkCUDA(__LINE__, cudaMemcpy( scene_type_gpu, scene_type_cpu, numel_batch_scene_classification * sizeofStorageT, cudaMemcpyHostToDevice) );
    checkCUDA(__LINE__, cudaMemcpy( noise_value_gpu, noise_value_cpu, numel_batch_noise * sizeofStorageT, cudaMemcpyHostToDevice) );
  };

  void forward(Phase phase_){
    //toc();
    //tic(); cout<<"wait lock ";
    lock.wait();
    epoch = epoch_prefetch;
    // cout << "swap!" << endl;
    //toc();
    //tic(); cout<<"Copy ";
    //tic(); cout<<"forward  ";
    // GPU_uint8_to_float_subtract(numel_batch_all_channel_crop, numel_all_channel_crop, dataGPU, meanGPU, out[0]->dataGPU);
    swap( out[0]->dataGPU, data_gpu);
    swap( out[1]->dataGPU, bbox_reg_val_gpu);
    swap( out[2]->dataGPU, bbox_reg_bin_gpu);
    swap( out[3]->dataGPU, bbox_cls_val_gpu);
    swap( out[4]->dataGPU, bbox_cls_bin_gpu);
    swap( out[5]->dataGPU, scene_type_gpu);
    swap( out[6]->dataGPU, noise_value_gpu);
    //toc();
    //checkCUDA(__LINE__, cudaMemcpy(out[0]->dataGPU, dataCPU->CPUmem,  dataCPU->numBytes() , cudaMemcpyHostToDevice) );
    //checkCUDA(__LINE__, cudaMemcpy(out[1]->dataGPU, labelCPU->CPUmem, labelCPU->numBytes(), cudaMemcpyHostToDevice) );
    //toc();
    //tic(); cout<<"Net ";
    lock = async( launch::async, &SceneHolisticOnlineTSDFLayer::prefetch, this);
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
    memoryBytes += 2 * out[1]->Malloc(bbox_regression_dims);
    cout << "MemoryBytes: " << memoryBytes << endl;

    out[2]->need_diff = false;
    memoryBytes += 2 * out[2]->Malloc(bbox_regression_dims);
    cout << "MemoryBytes: " << memoryBytes << endl;

    out[3]->need_diff = false;
    memoryBytes += 2 * out[3]->Malloc(bbox_classification_dims);
    cout << "MemoryBytes: " << memoryBytes << endl;

    out[4]->need_diff = false;
    memoryBytes += 2 * out[4]->Malloc(bbox_classification_c_dims);
    cout << "MemoryBytes: " << memoryBytes << endl;

    out[5]->need_diff = false;
    memoryBytes += 2 * out[5]->Malloc(scene_classification_dims);
    cout << "MemoryBytes: " << memoryBytes << endl;

    out[6]->need_diff = false;
    memoryBytes += 2 * out[6]->Malloc(noise_value_dims);
    cout << "MemoryBytes: " << memoryBytes << endl;

    // checkCUDA(__LINE__, cudaMalloc(&data_gpu, numel(data_dims) * sizeofVtype) );
    // checkCUDA(__LINE__, cudaMalloc(&bbox_value_gpu, numel(bbox_value_dims) * sizeofVtype) );
    // checkCUDA(__LINE__, cudaMalloc(&bbox_weight_gpu, numel(bbox_value_dims) * sizeofVtype) );
    // checkCUDA(__LINE__, cudaMalloc(&bbox_valid_gpu, numel(bbox_valid_dims) * sizeofVtype) );

    // // checkCUDA(__LINE__, cudaMalloc(&dataGPU, numel_batch_all_channel_crop) );
    // // memoryBytes += numel_batch_all_channel_crop;


    // // checkCUDA(__LINE__, cudaMalloc(&meanGPU, numel(meanCPU->dim) * sizeofVtype) );
    // // memoryBytes += meanCPU->numBytes();
    // // checkCUDA(__LINE__, cudaMemcpy( meanGPU,  meanCPU->CPUmem,  meanCPU->numBytes() , cudaMemcpyHostToDevice) );

    // // //tic();
    cout << "Memory allocated, read the first batch of data..." << endl;
    lock = async( launch::async, &SceneHolisticOnlineTSDFLayer::prefetch, this);

    return memoryBytes;
    // return 0;
  };  
};

class SceneHolisticOnlineTSDFSimpleLayer : public DataLayer {
public:
  int epoch_prefetch;

  string data_root;
  vector<string> data_list_name;
  vector<int> data_list_id;
  vector<int> label_list;
  // vector<int> label_list2;

  string file_list;
  string file_label;
  // string file_label2;

  string ground_truth_file;
  string camera_info_file;
  
  vector<int> grid_size;
  vector<float> spatial_range;
  // int GPU;
  
  int batch_size;
  future<void> lock;

  FILE* dataFILE;
  vector<size_t> ordering;

  DepthImage depth_image;
  vector<Scene3D> self_transformation;

  vector<int> data_dims;
  vector<int> scene_classification_dims;

  StorageT* data_gpu;
  StorageT* scene_type_cpu;
  StorageT* scene_type_gpu;
  // StorageT* scene_type_cpu2;
  // StorageT* scene_type_gpu2;

  int numel_batch_data;
  int numel_single_data;
  int numel_batch_scene_classification;
  int numel_single_scene_classification;

  bool shuffle_data;
  DataManager id_manager;

  int numofitems(){
    return data_list_name.size();
  };

  int numofitemsTruncated(){
    return batch_size * floor(double(numofitems())/double(batch_size));
  };

  void init(){

    // StorageT test[10];
    // memset(test, 0, 10*sizeofStorageT);
    // for (int i = 0; i < 10; ++i)
    // {
    //   std::cout << CPUStorage2ComputeT(test[i]) << std::endl;
    // }
    // cerr<<"Debug Stop"<<endl;
    // FatalError(__LINE__);


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
    // cout << "[0]: " << data_list_name[0] << endl;
    // cout << "[end]: " << data_list_name[num_data-1] << endl;
    cout << data_list_name.size() << "data load!" <<  endl;
    shuffle();
    counter = 0;

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

    // dataFILE = fopen( file_label2.c_str(), "r");
    // label_list2.clear();
    // fscanf( dataFILE, "%d", &num_data);
    // for ( int i=0; i<num_data; ++i )
    // {
    //   fscanf( dataFILE, "%d", &iBuffer);
    //   label_list2.push_back(iBuffer);
    // }
    // fclose(dataFILE);
    // cout << label_list2.size() << " label 2 loaded!" << endl;

    ////////////////////// read camera parameters
    cout << "Read camera parameter file ..." << endl;
    dataFILE = fopen( camera_info_file.c_str(), "rb");
    int checkNum;
    fread( (void*)(&checkNum), sizeof(unsigned int), 1, dataFILE);
    if (checkNum != num_data )
    {
      cerr<<"Data number inconsistent"<<endl;
      FatalError(__LINE__);
    }
    self_transformation.assign(checkNum, Scene3D());
    for (int sid=0; sid<checkNum; sid++)
    {
      self_transformation[sid].loadCameraInfo(dataFILE);
    }
    fclose(dataFILE);

    ////////////////////// Prepare for TSDF computing
    // depth_images.assign(1, DepthImage());
    float temp[10];
    temp[0] = spatial_range[0]; temp[1] = spatial_range[1];
    temp[2] = spatial_range[2]; temp[3] = spatial_range[3];
    temp[4] = spatial_range[4]; temp[5] = spatial_range[5];
    temp[6] = grid_size[3]; temp[7] = grid_size[2]; temp[8] = grid_size[1];
    temp[9] = 3.0 * (spatial_range[1] - spatial_range[0]) / grid_size[3];
    cout << "temp[9] = " << temp[9] << endl;
    depth_image.SceneSetting(temp);

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

    scene_classification_dims.clear();
    scene_classification_dims.push_back(batch_size);
    scene_classification_dims.push_back(1);
    scene_classification_dims.push_back(1);
    scene_classification_dims.push_back(1);
    scene_classification_dims.push_back(1);
    numel_batch_scene_classification = numel(scene_classification_dims);
    numel_single_scene_classification = sizeofitem(scene_classification_dims);
    cout << "scene_classification_dims: " << vecPrintString(scene_classification_dims) << endl;
    cout << "numel_batch_scene_classification: " << numel_batch_scene_classification << endl;
    cout << "numel_single_scene_classification: " << numel_single_scene_classification << endl;

    /////////////////// Allocate memory
    scene_type_cpu = new StorageT[numel_batch_scene_classification];
    // scene_type_cpu2 = new StorageT[numel_batch_scene_classification];

    checkCUDA(__LINE__, cudaMalloc(&data_gpu, numel_batch_data * sizeofStorageT) );
    checkCUDA(__LINE__, cudaMalloc(&scene_type_gpu, numel_batch_scene_classification * sizeofStorageT) );
	// checkCUDA(__LINE__, cudaMalloc(&scene_type_gpu2, numel_batch_scene_classification * sizeofStorageT) );

    id_manager.init( data_list_id, label_list, 8, rng , false, shuffle_data );

  };

  SceneHolisticOnlineTSDFSimpleLayer(string name_, Phase phase_): DataLayer(name_){
    phase = phase_;
    init();
  };

  SceneHolisticOnlineTSDFSimpleLayer(JSON* json){
    SetValue(json, name,    "Whatever")
    SetValue(json, phase,   Training)
    SetOrDie(json, camera_info_file )
    SetOrDie(json, file_list )
    SetOrDie(json, file_label )
    // SetOrDie(json, file_label2)

    SetOrDie(json, data_root )
    SetOrDie(json, grid_size )
    SetOrDie(json, spatial_range )
    SetValue(json, batch_size,  64)
    SetValue(json, shuffle_data, true)

    cout << "file_list:　" << file_list << endl;
    cout << "file_label: " << file_label << endl;
	cout << "file_label2: " << file_label << endl;

    cout << "camera_info_file: " << camera_info_file << endl;
    cout << "data_root: " << data_root << endl;

    cout << "grid_size: " << vecPrintString(grid_size) << endl;
    cout << "spatial_range: " << vecPrintString(spatial_range) << endl;
    cout << "batch_size: " << batch_size << endl;
    init();
  };

  ~SceneHolisticOnlineTSDFSimpleLayer(){
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

    // checkCUDA(__LINE__,cudaSetDevice(GPU));
    checkCUDA(__LINE__,cudaSetDevice(GPU));
    // tic(); cout<<"read disk  ";

    memset( (void*)scene_type_cpu, 0, sizeofStorageT * numel_batch_scene_classification);
    // memset( (void*)scene_type_cpu2, 0, sizeofStorageT * numel_batch_scene_classification);

    for (size_t i = 0; i < batch_size; ++i)
    {
      int local_id = id_manager.next_id(epoch_prefetch);
      // cout << "local_id: " << local_id << endl;

      string data_name = data_list_name[local_id];
      string data_file = data_root + data_name + "_depth.bin";    
      int data_id = data_list_id[local_id];

      ///////////// read data
      depth_image.readImage(data_file);
      // cout << "after depth" << endl;

      ///////////// compute TSDF
      // cout << "Enter prefetch: " << endl;
      // cout << "SetupMatrix" << endl;
      // depth_image.SetupMatrix(ground_truth[data_id].K, ground_truth[data_id].R, ground_truth[data_id].Rot, noise_tsl);
      depth_image.SetupMatrix(self_transformation[local_id].K, self_transformation[local_id].R, self_transformation[local_id].Rot, self_transformation[local_id].Tsl);
      // cout << "Transform Grid point" << endl;
      depth_image.transformGridPoints();
      // cout << "ComputeTSDF" << endl;
      // depth_image.ComputeTSDF();
      depth_image.ComputeTSDF_storage();
      // cout <<"prefetch end" <<  endl;

      checkCUDA(__LINE__, cudaMemcpy( data_gpu + i*numel_single_data,  depth_image.tsdf_gpu_storage,  numel_single_data * sizeofStorageT, cudaMemcpyDeviceToDevice) );
      // checkCUDA(__LINE__, cudaMemcpy( data_gpu + i*numel_single_data,  GPUCompute2StorageT((ComputeT*)depth_image.tsdf_gpu),  numel_single_data * sizeofStorageT, cudaMemcpyDeviceToDevice) );
      // GPUCompute2StorageT

      int label_id = label_list[local_id];
      // int label_id2 = label_list2[local_id];
      StorageT* scene_type_target = scene_type_cpu + i * numel_single_scene_classification;
      // StorageT* scene_type_target2 = scene_type_cpu2 + i * numel_single_scene_classification;
      // *scene_type_target = label_id;
      *scene_type_target = CPUCompute2StorageT( ComputeT(label_id) );
	  // *scene_type_target2 = CPUCompute2StorageT( ComputeT(label_id2) );
    }
    
    /////////////////// ship to gpu
    checkCUDA(__LINE__, cudaMemcpy( scene_type_gpu, scene_type_cpu, numel_batch_scene_classification * sizeofStorageT, cudaMemcpyHostToDevice) );
    // checkCUDA(__LINE__, cudaMemcpy( scene_type_gpu2, scene_type_cpu2, numel_batch_scene_classification * sizeofStorageT, cudaMemcpyHostToDevice) );
  };

  void forward(Phase phase_){
    //toc();
    //tic(); cout<<"wait lock ";
    lock.wait();
    epoch = epoch_prefetch;
    // cout << "swap!" << endl;
    //toc();
    //tic(); cout<<"Copy ";
    //tic(); cout<<"forward  ";
    // GPU_uint8_to_float_subtract(numel_batch_all_channel_crop, numel_all_channel_crop, dataGPU, meanGPU, out[0]->dataGPU);
    swap( out[0]->dataGPU, data_gpu);
    swap( out[1]->dataGPU, scene_type_gpu);
    // swap( out[2]->dataGPU, scene_type_gpu2);
    //toc();
    //checkCUDA(__LINE__, cudaMemcpy(out[0]->dataGPU, dataCPU->CPUmem,  dataCPU->numBytes() , cudaMemcpyHostToDevice) );
    //checkCUDA(__LINE__, cudaMemcpy(out[1]->dataGPU, labelCPU->CPUmem, labelCPU->numBytes(), cudaMemcpyHostToDevice) );
    //toc();
    //tic(); cout<<"Net ";
    lock = async( launch::async, &SceneHolisticOnlineTSDFSimpleLayer::prefetch, this);
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
    memoryBytes += 2 * out[1]->Malloc(scene_classification_dims);
    cout << "MemoryBytes: " << memoryBytes << endl;

    // out[2]->need_diff = false;
    // memoryBytes += 2 * out[2]->Malloc(scene_classification_dims);
    // cout << "MemoryBytes: " << memoryBytes << endl;

    // // //tic();
    cout << "Memory allocated, read the first batch of data..." << endl;
    lock = async( launch::async, &SceneHolisticOnlineTSDFSimpleLayer::prefetch, this);

    return memoryBytes;
    // return 0;
  };  
};

class SceneHolisticOnlineTSDFSimpleRegLayer : public DataLayer {
public:
  int epoch_prefetch;

  string data_root;
  vector<string> data_list_name;
  vector<int> data_list_id;
  vector<float> label_list;

  string file_list;
  string file_label;

  string ground_truth_file;
  string camera_info_file;
  
  vector<int> grid_size;
  vector<float> spatial_range;
  
  int batch_size;
  future<void> lock;

  FILE* dataFILE;
  vector<size_t> ordering;

  DepthImage depth_image;
  vector<Scene3D> self_transformation;

  vector<int> data_dims;
  vector<int> scene_classification_dims;

  StorageT* data_gpu;
  StorageT* scene_type_cpu;
  StorageT* scene_type_gpu;

  int numel_batch_data;
  int numel_single_data;
  int numel_batch_scene_classification;
  int numel_single_scene_classification;

  bool shuffle_data;
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
    float fBuffer;
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
    // cout << "[0]: " << data_list_name[0] << endl;
    // cout << "[end]: " << data_list_name[num_data-1] << endl;
    cout << data_list_name.size() << "data load!" <<  endl;
    shuffle();
    counter = 0;

    ////////////////////// read label list
    dataFILE = fopen( file_label.c_str(), "r");
    label_list.clear();
    fscanf( dataFILE, "%d", &num_data);
    for ( int i=0; i<3*num_data; ++i )
    {
      fscanf( dataFILE, "%f", &fBuffer);
      label_list.push_back(fBuffer);
    }
    fclose(dataFILE);
    cout << label_list.size() << " label loaded!" << endl;

    // dataFILE = fopen( file_label2.c_str(), "r");
    // label_list2.clear();
    // fscanf( dataFILE, "%d", &num_data);
    // for ( int i=0; i<num_data; ++i )
    // {
    //   fscanf( dataFILE, "%d", &iBuffer);
    //   label_list2.push_back(iBuffer);
    // }
    // fclose(dataFILE);
    // cout << label_list2.size() << " label 2 loaded!" << endl;

    ////////////////////// read camera parameters
    cout << "Read camera parameter file ..." << endl;
    dataFILE = fopen( camera_info_file.c_str(), "rb");
    int checkNum;
    fread( (void*)(&checkNum), sizeof(unsigned int), 1, dataFILE);
    if (checkNum != num_data )
    {
      cerr<<"Data number inconsistent"<<endl;
      FatalError(__LINE__);
    }
    self_transformation.assign(checkNum, Scene3D());
    for (int sid=0; sid<checkNum; sid++)
    {
      self_transformation[sid].loadCameraInfo(dataFILE);
    }
    fclose(dataFILE);

    ////////////////////// Prepare for TSDF computing
    // depth_images.assign(1, DepthImage());
    float temp[10];
    temp[0] = spatial_range[0]; temp[1] = spatial_range[1];
    temp[2] = spatial_range[2]; temp[3] = spatial_range[3];
    temp[4] = spatial_range[4]; temp[5] = spatial_range[5];
    temp[6] = grid_size[3]; temp[7] = grid_size[2]; temp[8] = grid_size[1];
    temp[9] = 3.0 * (spatial_range[1] - spatial_range[0]) / grid_size[3];
    cout << "temp[9] = " << temp[9] << endl;
    depth_image.SceneSetting(temp);

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

    scene_classification_dims.clear();
    scene_classification_dims.push_back(batch_size);
    scene_classification_dims.push_back(3);
    scene_classification_dims.push_back(1);
    scene_classification_dims.push_back(1);
    scene_classification_dims.push_back(1);
    numel_batch_scene_classification = numel(scene_classification_dims);
    numel_single_scene_classification = sizeofitem(scene_classification_dims);
    cout << "scene_classification_dims: " << vecPrintString(scene_classification_dims) << endl;
    cout << "numel_batch_scene_classification: " << numel_batch_scene_classification << endl;
    cout << "numel_single_scene_classification: " << numel_single_scene_classification << endl;

    /////////////////// Allocate memory
    scene_type_cpu = new StorageT[numel_batch_scene_classification];
    // scene_type_cpu2 = new StorageT[numel_batch_scene_classification];

    checkCUDA(__LINE__, cudaMalloc(&data_gpu, numel_batch_data * sizeofStorageT) );
    checkCUDA(__LINE__, cudaMalloc(&scene_type_gpu, numel_batch_scene_classification * sizeofStorageT) );
	// checkCUDA(__LINE__, cudaMalloc(&scene_type_gpu2, numel_batch_scene_classification * sizeofStorageT) );

    vector<int> vBuffer;
    vBuffer.assign(label_list.size()/3, 0);
    id_manager.init( data_list_id, vBuffer, 8, rng , false, shuffle_data );

  };

  SceneHolisticOnlineTSDFSimpleRegLayer(string name_, Phase phase_): DataLayer(name_){
    phase = phase_;
    init();
  };

  SceneHolisticOnlineTSDFSimpleRegLayer(JSON* json){
    SetValue(json, name,    "Whatever")
    SetValue(json, phase,   Training)
    SetOrDie(json, camera_info_file )
    SetOrDie(json, file_list )
    SetOrDie(json, file_label )
    // SetOrDie(json, file_label2)

    SetOrDie(json, data_root )
    SetOrDie(json, grid_size )
    SetOrDie(json, spatial_range )
    SetValue(json, batch_size,  64)
    SetValue(json, shuffle_data, true)

    cout << "file_list:　" << file_list << endl;
    cout << "file_label: " << file_label << endl;
	cout << "file_label2: " << file_label << endl;

    cout << "camera_info_file: " << camera_info_file << endl;
    cout << "data_root: " << data_root << endl;

    cout << "grid_size: " << vecPrintString(grid_size) << endl;
    cout << "spatial_range: " << vecPrintString(spatial_range) << endl;
    cout << "batch_size: " << batch_size << endl;
    init();
  };

  ~SceneHolisticOnlineTSDFSimpleRegLayer(){
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

    // checkCUDA(__LINE__,cudaSetDevice(GPU));
    checkCUDA(__LINE__,cudaSetDevice(GPU));
    // tic(); cout<<"read disk  ";

    memset( (void*)scene_type_cpu, 0, sizeofStorageT * numel_batch_scene_classification);
    // memset( (void*)scene_type_cpu2, 0, sizeofStorageT * numel_batch_scene_classification);

    for (size_t i = 0; i < batch_size; ++i)
    {
      int local_id = id_manager.next_id(epoch_prefetch);
      // cout << "local_id: " << local_id << endl;

      string data_name = data_list_name[local_id];
      string data_file = data_root + data_name + "_depth.bin";    
      int data_id = data_list_id[local_id];

      ///////////// read data
      depth_image.readImage(data_file);
      // cout << "after depth" << endl;

      ///////////// compute TSDF
      // cout << "Enter prefetch: " << endl;
      // cout << "SetupMatrix" << endl;
      // depth_image.SetupMatrix(ground_truth[data_id].K, ground_truth[data_id].R, ground_truth[data_id].Rot, noise_tsl);
      depth_image.SetupMatrix(self_transformation[local_id].K, self_transformation[local_id].R, self_transformation[local_id].Rot, self_transformation[local_id].Tsl);
      // cout << "Transform Grid point" << endl;
      depth_image.transformGridPoints();
      // cout << "ComputeTSDF" << endl;
      // depth_image.ComputeTSDF();
      depth_image.ComputeTSDF_storage();
      // cout <<"prefetch end" <<  endl;

      checkCUDA(__LINE__, cudaMemcpy( data_gpu + i*numel_single_data,  depth_image.tsdf_gpu_storage,  numel_single_data * sizeofStorageT, cudaMemcpyDeviceToDevice) );
      // checkCUDA(__LINE__, cudaMemcpy( data_gpu + i*numel_single_data,  GPUCompute2StorageT((ComputeT*)depth_image.tsdf_gpu),  numel_single_data * sizeofStorageT, cudaMemcpyDeviceToDevice) );
      // GPUCompute2StorageT

      // int label_id = label_list[local_id];
      // int label_id2 = label_list2[local_id];
      StorageT* scene_type_target = scene_type_cpu + i * numel_single_scene_classification;
      // StorageT* scene_type_target2 = scene_type_cpu2 + i * numel_single_scene_classification;
      // *scene_type_target = label_id;

      int memory_id = local_id * 3;
      *scene_type_target     = CPUCompute2StorageT( ComputeT(label_list[memory_id] ));
      *(scene_type_target+1) = CPUCompute2StorageT( ComputeT(label_list[memory_id+1] ));
      *(scene_type_target+2) = CPUCompute2StorageT( ComputeT(label_list[memory_id+2] ));

	  // *scene_type_target2 = CPUCompute2StorageT( ComputeT(label_id2) );
    }
    
    /////////////////// ship to gpu
    checkCUDA(__LINE__, cudaMemcpy( scene_type_gpu, scene_type_cpu, numel_batch_scene_classification * sizeofStorageT, cudaMemcpyHostToDevice) );
    // checkCUDA(__LINE__, cudaMemcpy( scene_type_gpu2, scene_type_cpu2, numel_batch_scene_classification * sizeofStorageT, cudaMemcpyHostToDevice) );
  };

  void forward(Phase phase_){
    //toc();
    //tic(); cout<<"wait lock ";
    lock.wait();
    epoch = epoch_prefetch;
    // cout << "swap!" << endl;
    //toc();
    //tic(); cout<<"Copy ";
    //tic(); cout<<"forward  ";
    // GPU_uint8_to_float_subtract(numel_batch_all_channel_crop, numel_all_channel_crop, dataGPU, meanGPU, out[0]->dataGPU);
    swap( out[0]->dataGPU, data_gpu);
    swap( out[1]->dataGPU, scene_type_gpu);
    // swap( out[2]->dataGPU, scene_type_gpu2);
    //toc();
    //checkCUDA(__LINE__, cudaMemcpy(out[0]->dataGPU, dataCPU->CPUmem,  dataCPU->numBytes() , cudaMemcpyHostToDevice) );
    //checkCUDA(__LINE__, cudaMemcpy(out[1]->dataGPU, labelCPU->CPUmem, labelCPU->numBytes(), cudaMemcpyHostToDevice) );
    //toc();
    //tic(); cout<<"Net ";
    lock = async( launch::async, &SceneHolisticOnlineTSDFSimpleRegLayer::prefetch, this);
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
    memoryBytes += 2 * out[1]->Malloc(scene_classification_dims);
    cout << "MemoryBytes: " << memoryBytes << endl;

    // out[2]->need_diff = false;
    // memoryBytes += 2 * out[2]->Malloc(scene_classification_dims);
    // cout << "MemoryBytes: " << memoryBytes << endl;

    // // //tic();
    cout << "Memory allocated, read the first batch of data..." << endl;
    lock = async( launch::async, &SceneHolisticOnlineTSDFSimpleRegLayer::prefetch, this);

    return memoryBytes;
    // return 0;
  }; 
};
 