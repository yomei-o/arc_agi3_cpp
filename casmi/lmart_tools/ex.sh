#!/bin/bash
# ex.sh <name> <lmart args...> : run lmart train in C:\prog\casmi\lmo, log to e_<name>.log, print RESULT lines
n=$1; shift
ssh -i ~/.claude/keys/ort_build_key yomei@192.168.6.14 "cd /d C:\prog\casmi\lmo && C:\prog\casmi\fwdmodel\lmart\lmart.exe train $* out=m_$n.txt > e_$n.log 2>&1 & type e_$n.log" | grep -a "RESULT\|best iteration\|gain:\|CONTROL" | sed "s/^/[$n] /"
