# Build Dpsim
FROM sogno/dpsim:dev AS DPSIM_BUILDER

RUN mkdir /app
WORKDIR /app

# Clone DPSIM without history
RUN git clone --depth 1 --branch v1.2.1 https://github.com/sogno-platform/dpsim.git

WORKDIR /app/dpsim/
RUN mkdir -p build && cd build && \
    cmake \
      -DCIMPP=ON \
      -DCGMES_Build=ON \
      -Dwith_villas=OFF \
      .. && \
    make -j$(nproc)


# Build Camnas
FROM docker.io/library/julia:1.12.1 AS CAMNAS_BUILDER

# Install build tools
RUN apt-get update -y
RUN apt-get install -y git gcc g++ make

# Copy artifacts from the previous stage
COPY --from=DPSIM_BUILDER /app/dpsim /app/dpsim

# Clone Camnas without history
#RUN git clone --depth 1 --branch build https://github.com/IgnoreWarnings/CAMNAS.jl.git
#RUN CP TO SolverPlugins
# Use local dev folder instead
COPY . /app/dpsim/dpsim/src/SolverPlugins

WORKDIR /app/dpsim/dpsim/src/SolverPlugins

# Instantiate packages
RUN julia --project=$(pwd) --eval="using Pkg;Pkg.instantiate()"

# Create shared library
RUN make -j$(nproc) camnasjl.so
RUN ls /app/dpsim/dpsim/src/SolverPlugins/../../include
# RUN ls /app/dpsim/dpsim/src/SolverPlugins/../../../include/dpsim
RUN make -j$(nproc) plugin.so

# Run dpsim with CAMNAS
#FROM sogno/dpsim:dev AS DPSIM_BUILDER
#COPY --from=DPSIM_BUILDER /app/dpsim /app/dpsim

WORKDIR /app/dpsim/examples/cxx/

# WORKDIR /app/
# # Fetch cim grid data
# RUN git clone https://github.com/n-eiling/cim-grid-data.git
# RUN mkdir -p dpsim/build/_deps/cim-data-src/WSCC-09/WSCC-09_RX
# RUN cp -R cim-grid-data/WSCC-09/WSCC-09_RX/ dpsim/build/_deps/cim-data-src/WSCC-09/WSCC-09_RX

# ENV JL_MNA_ALLOW_GPU=false \
#     CUDA_VISIBLE_DEVICES=3 \
#     JULIA_BINDIR=/usr/local/julia/bin \
#     LD_LIBRARY_PATH=/app/dpsim/dpsim/src/SolverPlugins/CAMNAS.jl:/app/dpsim/dpsim/src/SolverPlugins/CAMNAS.jl/CAMNASCompiled/lib

# RUN WSCC_9bus_mult_coupled \
#     -U "Plugin" \
#     -P "camnasjl"

