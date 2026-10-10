// KV 池满时选谁（docs/LOCAL_INFERENCE.md §E.4；WP3 拥有）。Scheduler（WP2）在 llama_process 返回 1 时按 internal.h 的顺序调用：
// 先 chooseIdleEviction 腾空闲序列，再减半 n_batch 重试，n_batch == 1 仍放不下才 chooseVictim 让一个在途请求失败（FLR_KV_FULL），
// 其余请求照常（llama-server 是所有在处理的槽位一起失败）。结果只看传进来的状态，一样的输入总是一样的结果。
#include "internal.h"

namespace flr {

int chooseIdleEviction(const std::vector<SequenceStatus>& all)
{
    // 非主优先、占得多的优先（一样多时先腾排在前面的）；主对话等子 Agent 时它的序列空闲，KV 回合结束已落盘，最后才腾它
    const SequenceStatus* best = nullptr;
    for (const SequenceStatus& status : all) {
        if (status.active || status.n_tokens <= 0)
            continue;
        if (!best || (best->main && !status.main) || (best->main == status.main && status.n_tokens > best->n_tokens))
            best = &status;
    }
    return best ? best->seq : -1;
}

int chooseVictim(const std::vector<SequenceStatus>& active)
{
    // 非主序列里 token 最多的（一样多时选更年轻的请求：进度少，丢掉的活少）；都没有（只有主对话在跑）就最年轻的请求
    const SequenceStatus* largest = nullptr;
    const SequenceStatus* youngest = nullptr;
    for (const SequenceStatus& status : active) {
        if (!status.active)
            continue;
        if (!status.main
            && (!largest || status.n_tokens > largest->n_tokens || (status.n_tokens == largest->n_tokens && status.serial > largest->serial)))
            largest = &status;
        if (!youngest || status.serial > youngest->serial)
            youngest = &status;
    }
    if (largest)
        return largest->seq;
    return youngest ? youngest->seq : -1;
}

} // namespace flr
