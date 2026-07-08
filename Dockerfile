# Build Dpsim
FROM sogno/dpsim:dev AS DPSIM_BUILDER

RUN mkdir /app
WORKDIR /app

# Clone DPSIM without history
RUN git clone --depth 1 --branch v1.2.1 https://github.com/sogno-platform/dpsim.git

WORKDIR /app/dpsim/
RUN mkdir -p build && cd build && \
    cmake \
      -DCIMPP=OFF \
      -DCGMES_Build=ON \
      -Dwith_villas=OFF \
      -DPSIM_BUILD_DOC=OFF \
      -DPSIM_BUILD_EXAMPLES=ON \
      .. && \
    make -j$(nproc)


# Build Camnas
FROM docker.io/library/julia:1.12.1 AS CAMNAS_BUILDER

# Install build tools
RUN apt-get update -y
RUN apt-get install -y git gcc g++ make

# Copy artifacts from the previous stage
COPY --from=DPSIM_BUILDER /app/dpsim /app/dpsim
COPY --from=DPSIM_BUILDER /lib /lib
COPY --from=DPSIM_BUILDER /lib64 /lib64
COPY --from=DPSIM_BUILDER /usr/local/lib /usr/local/lib
COPY --from=DPSIM_BUILDER /usr/local/lib64 /usr/local/lib64

# Add lib64 to linker path
RUN echo "/usr/local/lib64" | tee /etc/ld.so.conf.d/local-lib64.conf
RUN ldconfig

# Clone Camnas without history
WORKDIR /app/dpsim/dpsim/src/SolverPlugins/CAMNAS.jl
#RUN git clone --depth 1 --branch build https://github.com/IgnoreWarnings/CAMNAS.jl.git
# Use local dev folder instead
COPY . .

# Instantiate packages
RUN julia --project=$(pwd) --eval="using Pkg;Pkg.instantiate()"

# Create shared library
RUN make -j$(nproc) camnasjl.so
RUN ls /app/dpsim/dpsim/src/SolverPlugins/../../include
# RUN ls /app/dpsim/dpsim/src/SolverPlugins/../../../include/dpsim
RUN make -j$(nproc) plugin.so

# Fetch cim grid data
WORKDIR /app/
RUN git clone https://github.com/n-eiling/cim-grid-data.git
# Note: The dpsim import path is relative and not absolute to build (likely broken)
RUN mkdir -p /app/dpsim/build/dpsim/examples/cxx/build/_deps/cim-data-src/WSCC-09/WSCC-09_RX
RUN cp -R /app/cim-grid-data/WSCC-09/WSCC-09_RX/* /app/dpsim/build/dpsim/examples/cxx/build/_deps/cim-data-src/WSCC-09/WSCC-09_RX

WORKDIR /app/dpsim/build/dpsim/examples/cxx


# JL_MNA_ALLOW_GPU=false \
# CUDA_VISIBLE_DEVICES=3 \
# JULIA_BINDIR=/usr/local/julia/bin \
# LD_LIBRARY_PATH=/app/dpsim/dpsim/src/SolverPlugins/CAMNAS.jl:/app/dpsim/dpsim/src/SolverPlugins/CAMNAS.jl/CAMNASCompiled/lib:$LD_LIBRARY_PATH \
# ./WSCC_9bus_mult_coupled \
#     -U Plugin \
#     -P camnasjl

# ENV JL_MNA_ALLOW_GPU=false \
#     CUDA_VISIBLE_DEVICES=3 \
#     JULIA_BINDIR=/usr/local/julia/bin \
#     LD_LIBRARY_PATH=/app/dpsim/dpsim/src/SolverPlugins/CAMNAS.jl:/app/dpsim/dpsim/src/SolverPlugins/CAMNAS.jl/CAMNASCompiled/lib

# RUN WSCC_9bus_mult_coupled \
#     -U "Plugin" \
#     -P "camnasjl"
