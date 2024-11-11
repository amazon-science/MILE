
mkdir -p abo_raw_data
cd abo_raw_data

[ ! -f abo-mvr.csv.xz ] && \
 echo "Downloding listings" && \
 wget https://amazon-berkeley-objects.s3.amazonaws.com/benchmarks/abo-mvr.csv.xz .


[ ! -f abo-images-small.tar ] && \
wget https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-images-small.tar . 


[ ! -d images ] && \
tar -xvf abo-images-small.tar

[ ! -f abo-benchmark-material.tar ] && \
wget https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-benchmark-material.tar

[ ! -d abo-benchmark-material ] && \
tar -xvf abo-benchmark-material.tar

cd -