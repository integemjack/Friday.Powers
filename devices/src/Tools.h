// 给智能体的 MCP 工具（docs/devices-protocol.md §4；Friday 把这个 Power 的 /mcp 当成名叫 client 的 MCP 服务，工具是 client__devices 这样）。
//
// Friday 调 MCP 工具有 120 秒的上限：烧录、跑工具这种慢的做成后台任务（job），100 秒内做完直接回结果，
// 没做完先回 job_id 和进度，智能体用 job_wait 接着等。
#pragma once

#include "Bridge.h"
#include "Monitor.h"
#include "Ports.h"

#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace devices {

class Tools {
public:
    Tools(Bridge& bridge, Ports& ports);
    ~Tools();

    /// tools/list 的 tools
    json list() const;
    /// tools/call：返回 MCP 的 CallToolResult（{content: […], isError?}）
    json call(const std::string& name, const json& arguments);

private:
    struct Job;

    json devicesTool(const json& args);
    json serialOpen(const json& args);
    json tunnelOpen(const json& args);
    json androidAttach(const json& args);
    json iosAttach(const json& args);
    json portsTool(const json& args);
    json portClose(const json& args);
    json flash(const json& args);
    json espInfo(const json& args);
    json cameraSnapshot(const json& args);
    json runTool(const json& args);
    json stopTool(const json& args);
    json jobWait(const json& args);
    json serialMonitor(const json& args);
    json serialRead(const json& args);
    json serialWrite(const json& args);
    json serialReset(const json& args);
    json serialClose(const json& args);
    std::shared_ptr<Monitor> monitor(const json& args);

    /// 起一个后台任务，最多等 waitSeconds 秒：做完回结果，没做完回 job_id
    json startJob(const std::string& title, int waitSeconds, std::function<json(Job&)> work);
    json waitJob(const std::shared_ptr<Job>& job, int waitSeconds);
    /// 在设备清单里找这个 id 的显示名（找不到用 id）
    std::string deviceLabel(const ClientInfo& client, const std::string& deviceId);
    void onNotify(const std::string& client, const std::string& method, const json& params);

    Bridge& m_bridge;
    Ports& m_ports;
    std::mutex m_mutex;
    std::condition_variable m_wake;
    std::map<std::string, std::shared_ptr<Job>> m_jobs;
    int m_nextJob = 0;
    /// 后台跑着的工具（run_tool background）：client + runId → 状态（device.tool.exited 来了更新）
    std::map<std::string, json> m_toolRuns;
    /// 串口监视：id → 会话（输出缓冲在里面）
    std::map<int, std::shared_ptr<Monitor>> m_monitors;
    int m_nextMonitor = 0;
};

} // namespace devices
