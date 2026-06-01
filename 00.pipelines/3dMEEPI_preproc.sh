#!/usr/bin/env bash

# shellcheck source=./utils.sh
source $( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )/../utils.sh

# Check if there is input
[[ ( $# -eq 0 ) ]] && displayhelp $0 1

# Preparing the default values for variables
tmp=/tmp
debug=no
TEs="9.46 24.66 39.86"
degibbs=no
skip_bfc=no

### print input
printline=$( basename -- $0 )
echo "${printline}" "$@"
printcall="${printline} $*"
# Parsing required and optional variables with flags
# Also checking if a flag is the help request or the version
while [ ! -z "$1" ]
do
	case "$1" in
		-anat)		anat=$2;shift;;		# Any 3dMEEPI of an echo/acq/run series, the others will be found automatically. Better to run on normRO

		-TEs)		TEs="$2";shift;;	# Echo Times of the expected input. Must be specified within " "
		-tmp)		tmp=$2;shift;;		# Root folder for temporary files, where a script-specific folder will be created. If not in debug mode, the latter will be deleted at the end.
		-debug)		debug=yes;;			# Turn on debug mode.
		-degibbs)	degibbs=yes;;		# Run DeGibbs on echoavg files.
		-skip_bfc)	skip_bfc=yes;;		# DO NOT run N4BiasFieldCorrection on raw files.

		-h)			displayhelp $0;;	# Display this help.
		-v)			version $0;exit 0;;	# Display the version.
		*)			echo "Wrong flag: $1";displayhelp 1;; # Anything else is wrong!
	esac
	shift
done

# Check input
checkreqvar anat
checkoptvar TEs tmp debug

# Debug
[[ ${debug} == "yes" ]] && set -x && trap 'set +x' EXIT
[[ ${debug} == "no" ]] && trap '[ "${tmp}" != "/" ] && rm -rf ${tmp}' EXIT

### Remove nifti suffix
anat=$( removeniisfx ${anat} )

# Derived variables
scriptdir=$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )
anatname=$( basename ${anat} )

### Cath errors and exit on them
set -e
######################################
######### Script starts here #########
######################################
echo ""
echo "Make sure system python is used by prepending /usr/bin to PATH"
[[ "${PATH%%:*}" != "/usr/bin" ]] && export PATH=/usr/bin:$PATH
echo "PATH is set to $PATH"
echo ""

cwd=$(pwd)

# Parse anat filename
declare -A bids
extract_BIDS_entities "${anat}" bids echo

if_missing_do stop ${bids[root]}
if_missing_do mkdir ${bids[root]}/code/logs

# Create tmp folder
tmp="$(mktemp --tmpdir=${tmp} -d sub-${bids[sub]}_ses-${bids[ses]}_3dmeepipreproc.XXXXXX)"

# Preparing log folder and log file, removing the previous one
logfile=${bids[root]}/code/logs/${anatname}_log
replace_and touch ${logfile}

echo "************************************" >> ${logfile}

exec 3>&1 4>&2

exec 1>${logfile} 2>&1

version
date
echo ""
echo ${printcall}
echo ""
echo "PATH is set to $PATH"
checkreqvar anat
checkoptvar TEs tmp debug

echo "************************************"
echo "************************************"

echo ""
echo ""
echo "************************************"
echo "***    Parsed BIDS info ${anatname}"
echo "************************************"
echo ""
echo ""

checkoptvar bids

# Set various folders
adir=${bids[root]}/sub-${bids[sub]}/ses-${bids[ses]}/anat
aderivdir=${bids[root]}/derivatives/vessels/sub-${bids[sub]}/ses-${bids[ses]}/anat
rderivdir=${bids[root]}/derivatives/vessels/sub-${bids[sub]}/ses-${bids[ses]}/reg
regref=${bids[root]}/derivatives/vessels/sub-${bids[sub]}/ses-${bids[ses]}/reg/sub-${bids[sub]}_vesselref.nii.gz
anatprefix=sub-${bids[sub]}_ses-${bids[ses]}

# First return of variables discovered so far
checkoptvar scriptdir anatname adir aderivdir rderivdir regref anatprefix

# Now move to more interesting things
cd ${adir} || exit 1

# Create folders
if_missing_do mkdir ${aderivdir}
if_missing_do mkdir ${rderivdir}

[[ ! -d ${aderivdir} ]] && exit 2
[[ ! -d ${rderivdir} ]] && exit 2

# Crop and bias field correct anats
for anatfile in ${anatprefix}_*_${bids[suffix]}.nii.gz
do
	anatfile=$( basename $( removeniisfx ${anatfile} ) )

	[[ "$anatfile" =~ (.*)_echo-([^_]+) ]] && boxfile=${BASH_REMATCH[1]} && echo=${BASH_REMATCH[2]}

	echo ""
	echo ""
	echo "************************************"
	echo "***    Crop and correct bias field ${anatfile}"
	echo "************************************"
	echo ""
	echo ""

	## 01.Crop based on the first echo
	if [[ "${bids[echo]}" -eq 1 ]]
	then
		3dAutobox -extent_ijkord_to_file ${tmp}/${boxfile}_box ${anatfile}.nii.gz
	fi

	coords=$( awk '{ printf "%s %s ", $2, $3 - $2 + 1 }' ${tmp}/${boxfile}_box )
	fslroi ${anatfile}.nii.gz ${tmp}/${anatfile}_ab.nii.gz ${coords}
	## 02. Bias Field Correction with ANTs
	# 02.1. Truncate (0.01) for Bias Correction
	echo "Performing BFC on ${anatfile}"
	ImageMath 3 ${tmp}/${anatfile}_trunc.nii.gz TruncateImageIntensity ${tmp}/${anatfile}_ab.nii.gz 0.02 0.98 256
	
	anatpipesfx=trunc
	# 02.2. Bias Correction
	[[ "${skip_bfc}" == "no" ]] && N4BiasFieldCorrection -d 3 -i ${tmp}/${anatfile}_trunc.nii.gz -o ${tmp}/${anatfile}_bfc.nii.gz && anatpipesfx=bfc
done

# Prepare echoes averaging and T2* mapping
anatfiles=()

# Check all possible acqs and runs when needed 
mapfile -t acqs < <(find "${adir}" -type f -printf "%f\n" | grep "${bids[suffix]}" | grep -oP '_acq-\K[^_]+' | sort -u)

for a in "${acqs[@]}"
do
	workanat=${anatprefix}_acq-${a}

	mapfile -t runs < <(find "${adir}" -type f -printf "%f\n" | grep "${workanat}" | grep "${bids[suffix]}" | grep -oP '_run-\K[^_]+' | sort -u)

	if [ ${#runs[@]} -eq 0 ] || [[ -z "${runs[0]}" && ${#runs[@]} -eq 1 ]]
	then
		anatfiles+=("${workanat}")
	else
		for r in "${runs[@]}"
		do
			anatfiles+=("${workanat}_run-${r}")
		done
	fi
done

echo "************************************"
echo "***    Check variables"
echo "************************************"

echo "acqs are " "${acqs[@]}"
echo "runs are " "${runs[@]}"
echo "anatfiles are " "${anatfiles[@]}"

for i in "${!anatfiles[@]}"
do
	anatfile=${anatfiles[i]}


	echo ""
	echo ""
	echo "************************************"
	echo "***    Dealing with echoes of ${anatfile}"
	echo "************************************"
	echo ""
	echo ""

	echo ""
	echo "---------------------------"
	echo "MAKE SURE PYTHON IS CORRECT"
	echo "---------------------------"
	alias python=/usr/bin/python
	alias python3=/usr/bin/python3
	echo "Python: $( which python ) $( which python3 )"
	echo ""

	echo ""
	echo "--------------"
	echo "Running t2smap"
	echo "--------------"
	echo "Python: $( which python ) $( which python3 )"

	# T2* mapping and optimal combination
	t2smap -d ${tmp}/${anatfile}_echo-?_${bids[suffix]}_${anatpipesfx}.nii.gz --masktype none -e ${TEs} --out-dir ${tmp}/${anatfile}_TED
	fslmaths ${tmp}/${anatfile}_TED/desc-optcom_bold.nii.gz ${tmp}/${anatfile}_optcom_${bids[suffix]}.nii.gz -odt float
	fslmaths ${tmp}/${anatfile}_TED/T2starmap.nii.gz ${tmp}/${anatfile}_t2star_${bids[suffix]}.nii.gz -odt float

	# echo average
	3dMean -prefix ${tmp}/${anatfile}_echoavg_${bids[suffix]}.nii.gz ${tmp}/${anatfile}_echo-?_${bids[suffix]}_${anatpipesfx}.nii.gz

	# Option to run degibbs here
	if [[ ${degibbs} == "yes" ]]
	then
		# MRtrix3 suggests doing degibbs after dwidenoise though
		mrdegibbs ${tmp}/${anatfile}_echoavg_${bids[suffix]}.nii.gz ${tmp}/${anatfile}_echoavg_${bids[suffix]}_degibbs.nii.gz -force
		fslmaths ${tmp}/${anatfile}_echoavg_${bids[suffix]}_degibbs.nii.gz -sub ${tmp}/${anatfile}_echoavg_${bids[suffix]}.nii.gz \
				 ${tmp}/${anatfile}_echoavg_${bids[suffix]}_gibbsnoise.nii.gz
		mv ${tmp}/${anatfile}_echoavg_${bids[suffix]}.nii.gz ${tmp}/${anatfile}_echoavg_${bids[suffix]}_undegibbed.nii.gz
		mv ${tmp}/${anatfile}_echoavg_${bids[suffix]}_degibbs.nii.gz ${tmp}/${anatfile}_echoavg_${bids[suffix]}.nii.gz
	fi

	for imgtype in echoavg optcom t2star
	do
		imgfile=${tmp}/${anatfile}_${imgtype}_${bids[suffix]}.nii.gz
		x=$( fslval ${imgfile} dim1 ) 
		y=$( fslval ${imgfile} dim2 ) 
		z=$( fslval ${imgfile} dim3 )
		(( x*=2 ))
		(( y*=2 ))
		(( z*=2 ))
		ResampleImage 3 ${imgfile} ${tmp}/${anatfile}_${imgtype}_upsampled_${bids[suffix]}.nii.gz ${x}x${y}x${z} 1 0
	done

	# realign to vesselref (first file in input)
	echo ""
	echo ""
	echo "************************************"
	echo "***    Spatially coregister ${anatfile}"
	echo "************************************"
	echo ""
	echo ""
	if (( i == 0 ))
	then
		if_missing_do copy ${tmp}/${anatfile}_echoavg_${bids[suffix]}.nii.gz ${rderivdir}/sub-${bids[sub]}_vesselref_downsampled.nii.gz
		if_missing_do copy ${tmp}/${anatfile}_echoavg_upsampled_${bids[suffix]}.nii.gz ${regref}
		for metric in echoavg optcom t2star
		do
			if_missing_do copy ${tmp}/${anatfile}_${metric}_upsampled_${bids[suffix]}.nii.gz ${aderivdir}/${anatfile}_${metric}_${bids[suffix]}2vesselref.nii.gz
		done
	else
		for metric in echoavg optcom t2star
		do
			if grep -Fxq ${anatfile} ${scriptdir}/3dmeepi_discardlist
			then
				echo "  !!!  Skipping ${anatfile} ${metric} due to bad quality !!!"
				mv ${tmp}/${anatfile}_${metric}_upsampled_${bids[suffix]}.nii.gz ${aderivdir}/${anatfile}_${metric}_${bids[suffix]}_badquality.nii.gz
			else
				flirt -in ${tmp}/${anatfile}_${metric}_upsampled_${bids[suffix]} -ref ${regref} -cost normcorr -searchcost normcorr -dof 6 \
				-omat ${rderivdir}/${anatfile}2vesselref_fsl.mat -o ${aderivdir}/${anatfile}_${metric}_${bids[suffix]}2vesselref.nii.gz
			fi
		done
	fi
done


echo ""
echo ""
echo "************************************"
echo "***    Average ${anatname}"
echo "************************************"
echo ""
echo ""

# Average all echo averages, optcoms, and t2* maps
3dMean -prefix ${aderivdir}/00.${anatprefix}_${bids[suffix]}_imgavg_preprocessed.nii.gz ${aderivdir}/${anatprefix}_*_echoavg_${bids[suffix]}2vesselref.nii.gz
3dMean -prefix ${aderivdir}/00.${anatprefix}_${bids[suffix]}_optcom_preprocessed.nii.gz ${aderivdir}/${anatprefix}_*_optcom_${bids[suffix]}2vesselref.nii.gz
3dMean -prefix ${aderivdir}/00.${anatprefix}_${bids[suffix]}_t2star_preprocessed.nii.gz ${aderivdir}/${anatprefix}_*_t2star_${bids[suffix]}2vesselref.nii.gz

echo ""
echo ""
echo "************************************"
echo "***    Brain extract 00.${anatprefix}_${bids[suffix]}_imgavg_preprocessed"
echo "************************************"
echo ""
echo ""

brain_extract -nii ${aderivdir}/00.${anatprefix}_${bids[suffix]}_imgavg_preprocessed.nii.gz -method bet -tmp ${tmp} -nobrain
mv ${aderivdir}/00.${anatprefix}_${bids[suffix]}_imgavg_preprocessed_brain_mask.nii.gz ${aderivdir}/00.${anatprefix}_${bids[suffix]}_brain_mask.nii.gz

cd ${cwd}

# Final output of the preprocessing:
# ${aderivdir}/00.${anatprefix}_${anatsuffix}_imgavg_preprocessed.nii.gz
# ${aderivdir}/00.${anatprefix}_${anatsuffix}_optcom_preprocessed.nii.gz
# ${aderivdir}/00.${anatprefix}_${anatsuffix}_t2star_preprocessed.nii.gz
# ${aderivdir}/00.${anatprefix}_${anatsuffix}_mask.nii.gz

echo ""
echo ""
echo "************************************"
echo "***    Preproc completed!"
echo "************************************"
