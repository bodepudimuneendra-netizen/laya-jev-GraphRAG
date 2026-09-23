FROM postgres:16

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    postgresql-server-dev-16 \
    git \
    bison \
    flex \
    make \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Clone and build Apache AGE
RUN git clone https://github.com/apache/age.git /tmp/age \
    && cd /tmp/age \
    && git checkout PG16 \
    && make install

# Clone and build pgvector
RUN git clone --branch v0.7.0 https://github.com/pgvector/pgvector.git /tmp/pgvector \
    && cd /tmp/pgvector \
    && make \
    && make install

# Clean up build files to reduce image size
RUN rm -rf /tmp/age /tmp/pgvector \
    && apt-get purge -y build-essential postgresql-server-dev-16 git bison flex make gcc \
    && apt-get autoremove -y

# Configure PostgreSQL to load AGE by default
RUN echo "shared_preload_libraries = 'age'" >> /usr/share/postgresql/postgresql.conf.sample
