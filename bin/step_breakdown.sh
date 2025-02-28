#!/bin/bash
#
# Copyright 2025 The Physics-Next Project Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

declare -A baseline=(
  ["kinematics"]="16.45"
  ["com_pos"]="12.37"
  ["crb"]="27.91"
  ["factor_m"]="27.48"
  ["make_constraint"]="42.39"
  ["transmission"]="3.54"
  ["com_vel"]="9.38"
  ["passive"]="3.22"
  ["rne"]="16.75"
  ["fwd_actuation"]="3.93"
  ["xfrc_accumulate"]="6.81"
  ["solve_m"]="8.88"
  ["solve"]="61.57"
)

printf "%16s %16s %16s\n" "step name" "MJWarp ns" "MJX ns"
echo -e "\nfwd_position:"
for benchmark in kinematics com_pos crb factor_m make_constraint transmission
do
  mjw_val=`mjx-testspeed --function=$benchmark --is_sparse=False --mjcf=humanoid/humanoid.xml --batch_size=8192 --nefc_total=819200 | grep "Total time per step" | awk '{print $(NF - 1)}'`
  printf "%16s %16s %16s\n" $benchmark $mjw_val ${baseline[$benchmark]}
done

echo -e "\nfwd_velocity:"
for benchmark in com_vel passive rne
do
  mjw_val=`mjx-testspeed --function=$benchmark --is_sparse=False --mjcf=humanoid/humanoid.xml --batch_size=8192 --nefc_total=819200 | grep "Total time per step" | awk '{print $(NF - 1)}'`
  printf "%16s %16s %16s\n" $benchmark $mjw_val ${baseline[$benchmark]}
done

echo -e "\n"
for benchmark in fwd_actuation
do
  mjw_val=`mjx-testspeed --function=$benchmark --is_sparse=False --mjcf=humanoid/humanoid.xml --batch_size=8192 --nefc_total=819200 | grep "Total time per step" | awk '{print $(NF - 1)}'`
  printf "%16s %16s %16s\n" $benchmark $mjw_val ${baseline[$benchmark]}
done

echo -e "\nfwd_acceleration:"
for benchmark in xfrc_accumulate solve_m
do
  mjw_val=`mjx-testspeed --function=$benchmark --is_sparse=False --mjcf=humanoid/humanoid.xml --batch_size=8192 --nefc_total=819200 | grep "Total time per step" | awk '{print $(NF - 1)}'`
  printf "%16s %16s %16s\n" $benchmark $mjw_val ${baseline[$benchmark]}
done

# echo -e "\n"
# for benchmark in solve
# do
#   mjw_val=`mjx-testspeed --function=$benchmark --is_sparse=False --mjcf=humanoid/humanoid.xml --batch_size=8192 --nefc_total=819200 | grep "Total time per step" | awk '{print $(NF - 1)}'`
#   printf "%16s %16s %16s\n" $benchmark $mjw_val ${baseline[$benchmark]}
# done
