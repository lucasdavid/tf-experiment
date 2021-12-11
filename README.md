# TF Experiment

Tensorflow implementation of {EXPERIMENT_ID}.

## Summary

...

## Building and Running

Add the data to the `./data` folder (or create a symbolic link with `ln -s /path/to/dataset data`).

The project can be build and run a notebook server instance with the following commands:
```shell
$ docker build -t ldavid/{EXPERIMENT_ID} .
$
$ docker run -it --rm -p 8888:8888 \
    -v $(pwd)/src:/tf/src \
    -v $(pwd)/data/cached/.keras:/root/.keras \
    -v $(pwd)/notebooks:/tf/notebooks \
    -v $(pwd)/data:/tf/data \
    -v $(pwd)/logs:/tf/logs \
    -v $(pwd)/config:/tf/config \
    ldavid/{EXPERIMENT_ID}
```

To run experiments, simply use the `-d` option in `docker run -itd ...` and
call the scripts `docker exec -it {INSTANCE_ID} python {PATH_TO_SCRIPT}`.

For example:
```shell
$ docker run -itd --rm -p 8888:8888 -v $(pwd)/src:/tf/src -v $(pwd)/notebooks:/tf/notebooks -v $(pwd)/data:/tf/data -v $(pwd)/data/cached/.keras:/root/.keras -v $(pwd)/logs:/tf/logs -v $(pwd)/config:/tf/config ldavid/{EXPERIMENT_ID}
6290cd3658b4779df48f5586f0e8587a63838319e638d6ad88587654b537a0bd

$ docker exec -it 6290 python /tf/src/baseline.py with /tf/config/docker/baseline.yml -F /tf/logs/baseline
```
