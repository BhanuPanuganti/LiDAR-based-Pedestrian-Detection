#ifndef IOU3D_NMS_H
#define IOU3D_NMS_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

// Helper functions for 3D math
// Using 'inline' prevents the "already defined in iou3d_cpu.obj" error
inline int check_rect_cross(const float *a, const float *b, const float *c, const float *d);
inline float cross_product(const float *a, const float *b, const float *c);

int boxes_aligned_overlap_bev_gpu(at::Tensor boxes_a, at::Tensor boxes_b, at::Tensor ans_overlap);
int boxes_overlap_bev_gpu(at::Tensor boxes_a, at::Tensor boxes_b, at::Tensor ans_overlap);
int paired_boxes_overlap_bev_gpu(at::Tensor boxes_a, at::Tensor boxes_b, at::Tensor ans_overlap);
int boxes_iou_bev_gpu(at::Tensor boxes_a, at::Tensor boxes_b, at::Tensor ans_iou);
int nms_gpu(at::Tensor boxes, at::Tensor keep, float nms_overlap_thresh);
int nms_normal_gpu(at::Tensor boxes, at::Tensor keep, float nms_overlap_thresh);

#endif