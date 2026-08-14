// RUN: mlir-translate -verify-diagnostics -split-input-file -mlir-to-llvmir %s

// -----

llvm.func @tma_store_reduce_tile_override_addr_coord_tensor_size_mismatch(%src : !llvm.ptr<3>, %tma_desc : !llvm.ptr, %override_addr : !llvm.ptr<1>, %d0 : i32, %d1 : i32, %d2 : i32, %d3 : i32, %d4 : i32, %ts0 : i16, %ts1 : i16, %ts2 : i16, %ts3 : i16, %ts4 : i16, %lstrd0 : i32, %lstrd1 : i32, %lstrd2 : i32, %lstrd3 : i32, %ustrd : i16, %ch : i64) {
  // expected-error @below {{Expected coordinates size to be equal to tensor size}}
  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1] tensor_size[%ts0] {redKind = #nvvm.tma_redux_kind<add>, mode = #nvvm.tma_store_mode<tile>} : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>
  llvm.return
}

// -----

llvm.func @tma_store_reduce_tile_override_addr_lower_stride_tensor_size_mismatch(%src : !llvm.ptr<3>, %tma_desc : !llvm.ptr, %override_addr : !llvm.ptr<1>, %d0 : i32, %d1 : i32, %d2 : i32, %d3 : i32, %d4 : i32, %ts0 : i16, %ts1 : i16, %ts2 : i16, %ts3 : i16, %ts4 : i16, %lstrd0 : i32, %lstrd1 : i32, %lstrd2 : i32, %lstrd3 : i32, %ustrd : i16, %ch : i64) {
  // expected-error @below {{Expected lower_stride size to be equal to one less than tensor size}}
  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1] tensor_size[%ts0, %ts1] lower_stride[%lstrd0, %lstrd1] upper_stride[%ustrd] {redKind = #nvvm.tma_redux_kind<add>, mode = #nvvm.tma_store_mode<tile>} : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>
  llvm.return
}

// -----

llvm.func @tma_store_reduce_tile_override_addr_stride_mismatch(%src : !llvm.ptr<3>, %tma_desc : !llvm.ptr, %override_addr : !llvm.ptr<1>, %d0 : i32, %d1 : i32, %d2 : i32, %d3 : i32, %d4 : i32, %ts0 : i16, %ts1 : i16, %ts2 : i16, %ts3 : i16, %ts4 : i16, %lstrd0 : i32, %lstrd1 : i32, %lstrd2 : i32, %lstrd3 : i32, %ustrd : i16, %ch : i64) {
  // expected-error @below {{Expected lower_stride and upper_stride to be either both present or both absent}}
  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1] tensor_size[%ts0, %ts1] lower_stride[%lstrd0] {redKind = #nvvm.tma_redux_kind<add>, mode = #nvvm.tma_store_mode<tile>} : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>
  llvm.return
}

// -----

llvm.func @tma_store_reduce_tile_override_addr_im2col_dim_stride(%src : !llvm.ptr<3>, %tma_desc : !llvm.ptr, %override_addr : !llvm.ptr<1>, %d0 : i32, %d1 : i32, %d2 : i32, %d3 : i32, %d4 : i32, %ts0 : i16, %ts1 : i16, %ts2 : i16, %ts3 : i16, %ts4 : i16, %lstrd0 : i32, %lstrd1 : i32, %lstrd2 : i32, %lstrd3 : i32, %ustrd : i16, %ch : i64) {
  // expected-error @below {{Only tile mode supports override address with dim and stride}}
  nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2] tensor_size[%ts0, %ts1, %ts2] {redKind = #nvvm.tma_redux_kind<add>, mode = #nvvm.tma_store_mode<im2col>} : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>
  llvm.return
}

