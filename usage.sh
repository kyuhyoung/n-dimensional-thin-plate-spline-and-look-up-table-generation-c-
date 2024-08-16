make
if [ $? -eq 0 ]; then
	BASEDIR=${PWD##*/}
	#BASEDIR=$(cd $(dirname $0) && pwd)
	#echo "Script location: ${BASEDIR}"
	./${BASEDIR} 3
else
	echo Make FAIL
fi
