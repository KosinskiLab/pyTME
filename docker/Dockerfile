FROM python:3.11.5

RUN apt-get update \
	&& apt-get install -y libhdf5-dev \
	&& rm -rf /var/lib/apt/lists/*

COPY . /pyTME
RUN python3 -mpip --no-cache-dir install pyTME/ && rm -rf /pyTME

ENTRYPOINT ["bash"]