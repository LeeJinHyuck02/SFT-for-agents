#!/usr/bin/env bash
# run_train.sh / run_infer.sh 공용 함수. 직접 실행하지 않는다 (source 전용).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export UGRP_ROOT="$ROOT" PYTHONPATH="$ROOT/src" PYTHONIOENCODING=utf-8 TOKENIZERS_PARALLELISM=false
mkdir -p logs results

# 환경 구성 + 모델 다운로드. 같은 Pod에서 한 번 끝냈으면 건너뛴다 (학습 → 추론을 이어서 할 때).
# 완료 표시는 모델 캐시·pip 패키지와 같은 Container Disk(/root)에 둔다. /workspace(Volume Disk)에 두면
# Pod 재시작·Edit Pod로 Container Disk만 초기화됐을 때 표시만 남아 setup을 잘못 건너뛴다.
SETUP_MARKER="${SETUP_MARKER:-/root/.ugrp_setup_done}"
ensure_setup() {
  if [ -f "$SETUP_MARKER" ] && [ -z "${FORCE_SETUP:-}" ]; then
    echo "[SETUP] 이미 완료됨 (다시 하려면 FORCE_SETUP=1)"
  else
    bash scripts/pod_setup.sh
    touch "$SETUP_MARKER"
  fi
}

# detach <세션이름> <스크립트> <인자...> : tmux(없으면 nohup) 안에서 다시 실행하고 즉시 돌아온다.
detach() {
  local session="$1" script="$2"; shift 2
  local log="logs/${session}.log" cmd="bash $script"
  for arg in "$@"; do cmd+=" $(printf '%q' "$arg")"; done
  if command -v tmux >/dev/null; then
    if tmux has-session -t "$session" 2>/dev/null; then
      echo "[ABORT] tmux 세션 '$session'이 이미 있습니다. 실행 중인지 확인: tmux attach -t $session"
      echo "        끝난 세션이면: tmux kill-session -t $session"
      exit 1
    fi
    # HF_TOKEN은 선택 사항. 있으면 명시적으로 넘긴다: tmux 서버가 export 이전부터 떠 있었으면
    # 새 세션이 환경변수를 물려받지 못한다 (-e는 tmux 3.2+, 실패하면 옵션 없이 다시 시도)
    if [ -n "${HF_TOKEN:-}" ] && tmux new-session -d -s "$session" -e "HF_TOKEN=$HF_TOKEN" "$cmd 2>&1 | tee -a $log" 2>/dev/null; then
      :
    else
      tmux new-session -d -s "$session" "$cmd 2>&1 | tee -a $log"
    fi
    echo "tmux 세션 '$session'에서 실행 중.  보기: tail -f $log   /   tmux attach -t $session (나올 때 Ctrl+B, D)"
  else
    nohup bash -c "$cmd" >> "$log" 2>&1 &
    echo "nohup으로 실행 중 (PID $!).  보기: tail -f $log"
  fi
}

# run_stages <재개명령> <시작단계|""> <단계...> : 호출한 스크립트가 정의한 run_stage를 단계별로 실행한다.
run_stages() {
  local resume="$1" from="$2"; shift 2
  local skip=0; [ -n "$from" ] && skip=1
  for stage in "$@"; do
    [ "$stage" = "$from" ] && skip=0
    [ "$skip" -eq 1 ] && continue
    echo; echo "######## [$(date '+%F %T')] $stage 시작 ########"
    local start; start=$(date +%s)
    # if/! 문맥에서는 set -e가 꺼지므로, 단계 안의 어느 명령이든 실패하면 멈추도록 서브셸에서 실행한다
    set +e; ( set -e; run_stage "$stage" ); local rc=$?; set -e
    if [ "$rc" -ne 0 ]; then
      echo "######## $stage 실패. 원인 수정 후:  $resume --from $stage  ########"
      echo "$(date '+%F %T') FAILED $stage" >> logs/STATE
      exit 1
    fi
    echo "$(date '+%F %T') DONE $stage $(( $(date +%s) - start ))s" >> logs/STATE
  done
  if [ "$skip" -eq 1 ]; then echo "[ABORT] --from '$from' 은(는) 없는 단계입니다. 가능한 단계: $*"; exit 1; fi
}

print_download_notice() {
  echo
  echo "================================================================"
  echo " 완료. Pod를 끄기 전에 아래 파일을 반드시 내려받으십시오:"
  ls -lh "$ROOT"/results_*.tar.gz
  echo "================================================================"
}
