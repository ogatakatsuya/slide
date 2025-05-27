#!/bin/bash

mkdir -p src/md/$1
cp -r src/md/template/* src/md/$1

sed -i "s|{{PATH}}|$1|" src/md/$1/slide.md