docker_name=u_18_cpp_7_5_py_3_6_cv_4_5
#docker_name=u_20_cpp_9_4_py_3_9_cuda_11_3_cudnn_8_2_cv_4_7
#docker_name=u_22_cpp_11_3_py_3_10_cv_4_7
#docker_name=u_22
dir_cur=/workspace/${PWD##*/}
dir_data=/mnt/d/data
#: << 'END'
#################################################################################################
#   docker build
cd docker_file/; docker build --force-rm --shm-size=64g -t ${docker_name} -f Dockerfile_${docker_name} .; cd -
#END

#: << 'END'
#################################################################################################
#   docker info.
#docker run --rm -it -w $PWD -v $PWD:$PWD ${docker_name} bash docker_file/extract_docker_info.sh
docker run --rm -it -w $PWD -v $PWD:$PWD ${docker_name} sh -c ". ~/.bashrc && . ./extract_docker_info.sh"
#END

#: << 'END'
#################################################################################################
#   docker run
##	for SSH remote docker
docker run --rm -it --shm-size=64g --gpus '"device=0"' -e QT_DEBUG_PLUGINS=1 --net=host -v $HOME/.Xauthority:/root/.Xauthority:rw -e DISPLAY=$DISPLAY -w ${dir_cur} -v ${dir_data}:/data -v $PWD:${dir_cur} -v /etc/group:/etc/group:ro -v /etc/passwd:/etc/passwd:ro -v /etc/shadow:/etc/shadow:ro -v /etc/sudoers.d:/etc/sudoers.d:ro -v /tmp/.X11-unix:/tmp/.X11-unix:rw ${docker_name} fish
##	for local docker
# xhost +local:docker # in another terminal
#export DISPLAY=:0 && docker run --rm -it --shm-size=64g --gpus '"device=0"' -e QT_DEBUG_PLUGINS=1 -e DISPLAY=$DISPLAY -w ${dir_cur} -v ${dir_data}:/data -v $PWD:${dir_cur} -v /etc/group:/etc/group:ro -v /etc/passwd:/etc/passwd:ro -v /etc/shadow:/etc/shadow:ro -v /etc/sudoers.d:/etc/sudoers.d:ro -v /tmp/.X11-unix:/tmp/.X11-unix ${docker_name} fish



#END
