#!/bin/bash

for file in "$@"
do
    mv $file tke-$file
done
