#!/usr/bin/env bash

# shellcheck source=../utils.sh
source $( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )/../utils.sh

# Check if there is input
[[ ( $# -eq 0 ) ]] && displayhelp $0 1

# Preparing the default values for variables
voldiscard=10
polort=3
slicetimeinterp=none
despike=no
fdthr=.3
outthr=.05
den_motreg=no
den_detrend=no
applynuisance=no
fwhm=none
greyplot=no
tmp=/tmp
debug=no

### print input
printline=$( basename -- $0 )
echo "${printline}" "$@"
printcall="${printline} $*"
# Parsing required and optional variables with flags
# Also checking if a flag is the help request or the version
while [ ! -z "$1" ]
do
	case "$1" in
		-func)				func=$2;shift;;

		-voldiscard)		voldiscard=$2;shift;;
		-polort)			polort=$2;shift;;
		-slicetimeinterp)	slicetimeinterp=$2;shift;;
		-despike)			despike=yes;;
		-fdthr)				fdthr=$2;shift;;		# Censor FD threshold. This DOES NOT apply censoring automatically.
		-outthr)			outthr=$2;shift;;		# Censor outcount threshold. This DOES NOT apply censoring automatically.
		-den_motreg)		den_motreg=yes;;
		-den_detrend)		den_detrend=yes;;
		-applynuisance)		applynuisance=yes;;
		-make_greyplots)	greyplot=yes;;
		-fwhm)				fwhm=$2;shift;;

		-tmp)				tmp=$2;shift;;
		-debug)				debug=yes;;

		-h)			displayhelp $0;;
		-v)			version;exit 0;;
		*)			echo "Wrong flag: $1";displayhelp $0 1;;
	esac
	shift
done

# Check input
checkreqvar func
checkoptvar voldiscard polort slicetimeinterp despike fdthr \
			den_motreg den_detrend applynuisance greyplot fwhm tmp debug

# Debug
[[ ${debug} == "yes" ]] && set -x && trap 'set +x' EXIT
[[ ${debug} == "no" ]] && trap '[ -n "${tmp}" ] && [ "${tmp}" != "/" ] && rm -rf ${tmp}' EXIT
### Remove nifti suffix
func=$( removeniisfx ${func} )

# Derived variables
scriptdir=$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )
funcname=$( basename ${func} )

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

# Parse func filename
declare -A bids
extract_BIDS_entities "${func}" bids run

if_missing_do stop ${bids[root]}
if_missing_do mkdir ${bids[root]}/code/logs

# Create tmp folder
tmp="$(mktemp --tmpdir=${tmp} -d sub-${bids[sub]}_ses-${bids[ses]}_funcpreproc.XXXXXX)"

# Preparing log folder and log file, removing the previous one
logfile=${bids[root]}/code/logs/${funcname}_log
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
checkreqvar func
checkoptvar voldiscard polort slicetimeinterp despike fdthr \
			den_motreg den_detrend applynuisance greyplot fwhm tmp debug

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

fdir=${bids[root]}/sub-${bids[sub]}/ses-${bids[ses]}/func
fmapdir=${bids[root]}/sub-${bids[sub]}/ses-${bids[ses]}/fmap
fderivdir=${bids[root]}/derivatives/35msphys/sub-${bids[sub]}/ses-${bids[ses]}/func

checkoptvar fdir fderivdir fmapdir


if_missing_do mkdir ${fderivdir}

if [ -d ${fmapdir} ]
then
	echo "************************************"
	echo "*** Compute Pepolar ${funcname}"
	echo "************************************"
	echo "************************************"

	fmapfiles=()
	for fmap in "${fmapdir}"/*_dir-*.nii.gz
	do
		fmapname=$( basename $( removeniisfx ${fmap} ) )
		ImageMath 3 ${tmp}/${fmapname}_trunc.nii.gz TruncateImageIntensity ${fmapdir}/${fmapname}.nii.gz 0.02 0.98 256
		brain_extract -nii ${tmp}/${fmapname}_trunc -method bet -tmp ${tmp} -slice
		fmapfiles+=(${tmp}/${fmapname}_trunc_brain)
	done

	if [ ${#fmapfiles[@]} -gt 0 ] || [[ ! -z "${fmapfiles[0]}" && ${#fmapfiles[@]} -gt 1 ]]
	then
		pepolardir=${fderivdir}/${funcname%_run-*}_topup

		fslmerge -t ${pepolardir}/mgdmap "${fmapfiles[@]}"

		cd ${pepolardir}
		echo "Computing PEpolar map for ${func}"
		topup --imain=mgdmap --datain=${scriptdir}/acqparam.txt --out=outtp --verbose
		cd ${cwd}
	fi
fi

funcfiles=()
mapfile -t runs < <(find "${fdir}" -type f -printf "%f\n" | grep "task-${bids[task]}" | grep -oP '_run-\K[^_]+' | sort -u)

if [ ${#runs[@]} -eq 0 ] || [[ -z "${runs[0]}" && ${#runs[@]} -eq 1 ]]
then
	funcfiles+=("${func}")
else
	for r in "${runs[@]}"
	do
		funcfiles+=("${fdir}/${funcname%_run-*}_run-${r}_${bids[filesuffix]}")
	done
fi

mref=""
mask=""
for funcfile in ${funcfiles[@]}
do
	funcname=$( basename ${funcfile} )
	funcprefix=${funcname%"_${bids[filesuffix]}"}
	echo "************************************"
	echo "*** Func correct ${funcname}"
	echo "************************************"
	echo "************************************"

	nTR=$(fslval ${funcfile} dim4)

	funcsource=${funcfile}
	if [[ "${nTR}" -gt "1" ]]
	then
		# Separate noise volumes #!# Check if necessary
		(( noisevols=nTR-2 ))
		fslroi ${func} ${fderivdir}/${funcprefix}_gaussiannoise ${noisevols} -1

		# discard volumes
		(( endvol=nTR-2-voldiscard ))
		[[ "${voldiscard}" -gt "0" ]] && fslroi ${func} ${tmp}/${funcprefix}_dsd ${voldiscard} ${endvol}
		funcsource=${tmp}/${funcprefix}_dsd
	fi

	if [[ -z ${mref} ]]
	then
		# create mask & mref - this should trigger only on first run
		fslmaths ${funcsource} -Tmean ${tmp}/${funcprefix}_avg
		masksource=${tmp}/${funcprefix}_avg

		[ -e ${pepolardir}/outtp ] && applytopup --imain=${tmp}/${funcprefix}_avg --datain=${scriptdir}/acqparam.txt --inindex=1 \
				   --topup=${pepolardir}/outtp --out=${tmp}/${funcprefix}_avg_tpp --verbose --method=jac && masksource=${tmp}/${funcprefix}_avg_tpp

		ImageMath 3 ${tmp}/${funcprefix}_avg_trunc.nii.gz TruncateImageIntensity ${masksource}.nii.gz 0.02 0.98 256
		# For some reason ImageMath changes 3D grid of image, so 3dcalc puts it back in the right place.
		3dcalc -a ${masksource}.nii.gz -b ${tmp}/${funcprefix}_avg_trunc.nii.gz -expr "astep(a,0)*b" \
			   -prefix ${tmp}/${funcprefix}_avg_trunc.nii.gz -overwrite		

		brain_extract -nii ${tmp}/${funcprefix}_avg_trunc -method bet -tmp ${tmp} -slice
		mref=${fderivdir}/${funcprefix%_run-*}_brain
		mask=${mref}_mask
		mv ${tmp}/${funcprefix}_avg_trunc_brain.nii.gz ${mref}.nii.gz
		mv ${tmp}/${funcprefix}_avg_trunc_brain_mask.nii.gz ${mask}.nii.gz

	fi

	[[ ${nTR} -gt 1 ]] && 3dToutcount -mask ${mask}.nii.gz -fraction -polort 5 -legendre ${funcsource}.nii.gz > ${fderivdir}/${funcprefix}_outcount.1D

	[[ "${despike}" == "yes" ]] && echo "Despike ${funcname}" && 3dDespike -prefix ${tmp}/${funcprefix}_dsk.nii.gz ${funcsource}.nii.gz && funcsource=${tmp}/${funcprefix}_dsk

	if [[ "${slicetimeinterp}" != "none" ]]
	then
		echo "Slice Interpolate ${funcname}"
		3dTshift -Fourier -prefix ${tmp}/${funcprefix}_si.nii.gz \
				-tpattern ${slicetimeinterp} -overwrite \
				${funcsource}.nii.gz
		funcsource=${tmp}/${funcprefix}_si
	fi


	if [[ ${nTR} -gt 1 ]]
	then
		[[ ${greyplot} == "yes" ]] && echo "Create Greyplot ${funcname} premotion" \
								   && 3dGrayplot -input ${funcsource}.nii.gz -mask ${mask}.nii.gz \
												 -prefix ${fderivdir}/${funcprefix}_gp_pre.png -dimen 1800 1200 \
												 -polort ${polort} -peelorder -percent -range 3

		echo "************************************"
		echo "*** Func spacecomp ${funcname}"
		echo "************************************"
		echo "************************************"

		compute_fddvars -in ${funcsource}.nii.gz -m ${mask}.nii.gz
		mv ${funcsource}_dvars.par ${fderivdir}/${funcprefix}_dvars_pre.par

		mcflirt -in ${funcsource} -r ${mref} -out ${tmp}/${funcprefix}_mcf -stats -plots
		funcsource=${tmp}/${funcprefix}_mcf

		1d_tool.py -infile ${funcsource}.par -demean -write ${fderivdir}/${funcprefix}_mcf_demean.par -overwrite
		1d_tool.py -infile ${fderivdir}/${funcprefix}_mcf_demean.par -derivative -demean -write ${fderivdir}/${funcprefix}_mcf_deriv1.par -overwrite
		compute_fddvars -in ${funcsource}.nii.gz -m ${mask}.nii.gz
		mv ${funcsource}_dvars.par ${fderivdir}/${funcprefix}_dvars_post.par
		mv ${tmp}/${funcprefix}_fd.par ${fderivdir}/${funcprefix}_fd.par

		[[ ${greyplot} == "yes" ]] && echo "Create Greyplot ${funcname} postmotion" \
								   && 3dGrayplot -input ${funcsource}.nii.gz -mask ${mask}.nii.gz \
												 -prefix ${fderivdir}/${funcprefix}_gp_post.png -dimen 1800 1200 \
												 -polort ${polort} -peelorder -percent -range 3

		echo "************************************"
		echo "*** Func Nuiscomp ${funcname}"
		echo "************************************"
		echo "************************************"

		echo "Preparing censoring"
		1deval -a ${fderivdir}/${funcprefix}_fd.par -b=${fdthr} -c ${fderivdir}/${funcprefix}_outcount.1D -d=${outthr} -expr 'isnegative(a-b)*isnegative(c-d)' > ${fderivdir}/${funcprefix}_censor.1D

		# 04.2. Create matrix
		echo "Preparing nuisance matrix"

		run3dDeconvolve="3dDeconvolve -input ${funcsource}.nii.gz -float \
		-x1D ${fderivdir}/${funcprefix}_nuisreg_mat.1D \
		-xjpeg ${fderivdir}/${funcprefix}_nuisreg_mat.jpg \
		-x1D_stop"

		[[ "${den_detrend}" == "yes" ]] && run3dDeconvolve="${run3dDeconvolve} -polort ${polort}"
		[[ "${den_motreg}" == "yes" ]] && run3dDeconvolve="${run3dDeconvolve} -ortvec ${fderivdir}/${funcprefix}_mcf_demean.par motdemean \
														   -ortvec ${fderivdir}/${funcprefix}_mcf_deriv1.par motderiv1"

		echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++"
		echo "# Running 3dDeconvolve with the following parameters:"
		echo "   + Denoise motion regressors:        ${den_motreg}"
		echo "   + Denoise legendre polynomials:     ${den_detrend}"
		echo ""
		echo "# Generating the command:"
		echo ""
		echo "${run3dDeconvolve}"
		echo ""
		echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++"

		eval ${run3dDeconvolve}

		if [[ "${applynuisance}" == "yes" ]]
		then
			echo "Actually applying nuisance"
			fslmaths ${funcsource} -Tmean ${tmp}/${funcprefix}_avgfornuisance
			3dTproject -polort 0 -input ${funcsource}.nii.gz  -mask ${mask}.nii.gz \
			-ort ${fderivdir}/${funcprefix}_nuisreg_mat.1D -prefix ${tmp}/${funcprefix}_prj.nii.gz \
			-overwrite
			fslmaths ${tmp}/${funcprefix}_prj -add ${tmp}/${funcprefix}_avgfornuisance ${tmp}/${funcprefix}_den
			funcsource=${tmp}/${funcprefix}_den
		fi
	fi

	echo "************************************"
	echo "*** Apply Pepolar ${funcname}"
	echo "************************************"
	echo "************************************"

	[ -e ${pepolardir}/outtp ] && applytopup --imain=${funcsource} --datain=${scriptdir}/acqparam.txt --inindex=1 \
			   --topup=${pepolardir}/outtp --out=${tmp}/${funcprefix}_tpp --verbose --method=jac && funcsource=${tmp}/${funcprefix}_tpp

	if [[ ${fwhm} != "none" ]]
	then

		echo "************************************"
		echo "*** Func smoothing ${task} BOLD"
		echo "************************************"
		echo "************************************"

		3dBlurInMask -input ${funcsource}.nii.gz -prefix ${tmp}/${funcprefix}_sm.nii.gz -preserve -FWHM ${fwhm} -overwrite
		funcsource=${tmp}/${funcprefix}_sm
	fi

	3dcalc -a ${funcsource}.nii.gz -b ${mask}.nii.gz -expr 'a*b' \
		   -prefix ${fderivdir}/00.${funcprefix}_native_preprocessed.nii.gz \
		   -short -gscale -overwrite

	[[ ${greyplot} == "yes" && ${nTR} -gt 1 ]] && echo "Create Greyplot ${funcname} final" \
											   && 3dGrayplot -input ${funcsource}.nii.gz -mask ${mask}.nii.gz \
															 -prefix ${fderivdir}/${funcprefix}_gp_preprocessed.png -dimen 1800 1200 \
															 -polort ${polort} -peelorder -percent -range 3

done

cd ${cwd}