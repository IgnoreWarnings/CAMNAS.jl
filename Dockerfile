FROM alpine/git AS GIT

RUN mkdir /app
WORKDIR /app

# Clone Camnas without history
# RUN git clone --depth 1 https://github.com/IgnoreWarnings/CAMNAS.jl.git
COPY ../CAMNAS.jl /app/

# Clone DPSIM without history
RUN git clone --depth 1 --branch v1.2.1 https://github.com/sogno-platform/dpsim.git

# Fetch cim grid data
RUN git clone https://github.com/n-eiling/cim-grid-data.git
#RUN cp cim-grid-data/9bus-add-full/WSCC-09/WSCC-09_RX/ dpsim/build/_deps/cim-data-src/WSCC-09/WSCC-09_RX


# Build Camnas
FROM docker.io/library/julia:1.12.1 AS CAMNAS_BUILDER

# Copy artifacts from the previous stage
COPY --from=GIT /app /app
WORKDIR /app/CAMNAS.jl

# Install make
RUN apt-get update -y
RUN apt-get install -y gcc make

# Instantiate packages
RUN julia --project=$(pwd) --eval="using Pkg;Pkg.instantiate()"

# Create shared library
RUN make -j 4


# Build Dpsim
FROM sogno/dpsim:dev AS DPSIM_BUILDER

# Copy artifacts from the previous stage
COPY --from=CAMNAS_BUILDER /app /app

WORKDIR /app/dpsim/
RUN mkdir build && cd build && \
    cmake \
      -DCIMPP=ON \
      -DCGMES_Build=ON \
      -Dwith_villas=OFF \
      .. && \
    make -j$(nproc)


# Run example with CAMNAS
FROM docker.io/library/julia:1.12.6

# Copy artifacts from the previous stage
COPY --from=DPSIM_BUILDER /app /app

# Copy Plugin
RUN cp -R ../CAMNAS.jl ./dpsim/src/SolverPlugins/

WORKDIR /app/dpsim/examples/cxx/

ENV JL_MNA_ALLOW_GPU=false \
    CUDA_VISIBLE_DEVICES=3 \
    JULIA_BINDIR=/usr/local/julia/bin \
    LD_LIBRARY_PATH=/app/CAMNAS.jl:/app/CAMNAS.jl/CAMNASCompiled/lib

RUN WSCC_9bus_mult_coupled \
    -U "Plugin" \
    -P "camnasjl"

# RUN  JL_MNA_ALLOW_GPU="false" CUDA_VISIBLE_DEVICES=3 JULIA_BINDIR="/home/bauer/.juliaup/bin" \
#     LD_LIBRARY_PATH="/home/bauer/.julia/juliaup/julia-1.12.1+0.x64.linux.gnu/lib/julia/:/home/bauer/Codebase/dpsim/dpsim/src/SolverPlugins/CAMNAS.jl/:/home/bauer/Codebase/dpsim/dpsim/src/SolverPlugins/CAMNAS.jl/CAMNASCompiled/lib/" \
#     /home/bauer/Codebase/dpsim/build/dpsim/examples/cxx/WSCC_9bus_mult_coupled \
#     -U "Plugin" \
#     -P "camnasjl"

