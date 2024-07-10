#!/bin/bash
IFS='/'
read -ra str_array <<< "$1"
fn="${str_array[1]}"
IFS='.'
read -ra str_array <<< "$fn"
fn="${str_array[0]}"
echo $fn