// 一条推理线程排队用 GPU：定稿优先；实时字幕每路流只留最新的一个（旧的直接丢，算了也没用）。
#pragma once

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <thread>

namespace voice {

class InferenceQueue {
public:
    InferenceQueue() : m_thread([this] { run(); }) {}
    ~InferenceQueue()
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_stopping = true;
        }
        m_wake.notify_all();
        m_thread.join();
    }
    InferenceQueue(const InferenceQueue&) = delete;
    InferenceQueue& operator=(const InferenceQueue&) = delete;

    void submitFinal(std::function<void()> job)
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_finals.push_back(std::move(job));
        }
        m_wake.notify_one();
    }

    /// 同一个 key（一路流）只留最新的一个
    void submitPartial(uint64_t key, std::function<void()> job)
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_partials.find(key) == m_partials.end())
                m_order.push_back(key);
            m_partials[key] = std::move(job);
        }
        m_wake.notify_one();
    }

    void cancelPartial(uint64_t key)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_partials.erase(key);
    }

    size_t pending()
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_finals.size() + m_partials.size();
    }

private:
    void run()
    {
        for (;;) {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lock(m_mutex);
                m_wake.wait(lock, [this] { return m_stopping || !m_finals.empty() || !m_partials.empty(); });
                if (m_stopping)
                    return;
                if (!m_finals.empty()) {
                    job = std::move(m_finals.front());
                    m_finals.pop_front();
                } else {
                    while (!m_order.empty()) {
                        const uint64_t key = m_order.front();
                        m_order.pop_front();
                        const auto it = m_partials.find(key);
                        if (it != m_partials.end()) {
                            job = std::move(it->second);
                            m_partials.erase(it);
                            break;
                        }
                    }
                }
            }
            if (job)
                job();
        }
    }

    std::mutex m_mutex;
    std::condition_variable m_wake;
    std::deque<std::function<void()>> m_finals;
    std::map<uint64_t, std::function<void()>> m_partials;
    std::deque<uint64_t> m_order;
    bool m_stopping = false;
    std::thread m_thread;
};

} // namespace voice
