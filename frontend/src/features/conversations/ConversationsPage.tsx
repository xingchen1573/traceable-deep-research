import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { RefreshCw, MessageSquare, Trash2 } from 'lucide-react';
import { PageHeader } from '@/components/layout/PageHeader';
import { ErrorPanel } from '@/components/data/ErrorPanel';
import { Button } from '@/components/controls/Button';
import { SearchInput } from '@/components/controls/FormControls';
import { useConversations, useDeleteConversation } from '@/hooks';
import { formatDateTime, formatRelativeTime, formatNumber } from '@/utils/formatters';

export function ConversationsPage() {
  const navigate = useNavigate();
  const { data, isLoading, error, refetch } = useConversations();
  const deleteConv = useDeleteConversation();
  const [search, setSearch] = useState('');

  const filtered = data?.conversations?.filter((c) => {
    if (!search) return true;
    const q = search.toLowerCase();
    return (
      c.thread_id.toLowerCase().includes(q) ||
      c.title.toLowerCase().includes(q)
    );
  });

  const handleDelete = (e: React.MouseEvent, threadId: string) => {
    e.stopPropagation();
    if (confirm('确定删除该对话记录？此操作不可恢复。')) {
      deleteConv.mutate(threadId);
    }
  };

  return (
    <div className="space-y-4">
      <PageHeader
        title="对话历史"
        description="查看所有历史对话记录"
        actions={
          <Button variant="secondary" onClick={() => refetch()} icon={undefined}>
            <RefreshCw className="mr-1.5 h-4 w-4" />
            刷新
          </Button>
        }
      />

      {error && <ErrorPanel error={error} onRetry={() => refetch()} />}

      <SearchInput
        placeholder="搜索对话标识或标题"
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        wrapperClassName="max-w-sm"
      />

      {isLoading ? (
        <div className="rounded-panel border border-border bg-white p-8 text-center text-secondary">
          加载中...
        </div>
      ) : !filtered?.length ? (
        <div className="rounded-panel border border-border bg-white p-8 text-center text-secondary">
          <MessageSquare className="mx-auto mb-2 h-8 w-8 text-secondary/50" />
          <p>{search ? '没有匹配的对话' : '暂无对话记录'}</p>
          <p className="mt-1 text-xs">通过聊天接口发送消息后，对话记录将自动保存在此处</p>
        </div>
      ) : (
        <div className="overflow-x-auto rounded-panel border border-border bg-white">
          <table className="w-full text-left text-sm">
            <thead className="bg-page text-xs font-medium text-secondary">
              <tr>
                <th className="whitespace-nowrap px-4 py-3">对话</th>
                <th className="whitespace-nowrap px-4 py-3" style={{ width: '80px' }}>消息数</th>
                <th className="whitespace-nowrap px-4 py-3" style={{ width: '150px' }}>创建时间</th>
                <th className="whitespace-nowrap px-4 py-3" style={{ width: '120px' }}>最后活跃</th>
                <th className="whitespace-nowrap px-4 py-3" style={{ width: '60px' }}></th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {filtered.map((conv) => (
                <tr
                  key={conv.thread_id}
                  onClick={() => navigate(`/conversations/${encodeURIComponent(conv.thread_id)}`)}
                  className="cursor-pointer bg-white hover:bg-page"
                >
                  <td className="px-4 py-3">
                    <div>
                      <p className="font-medium text-primary">{conv.title || conv.thread_id}</p>
                      <p className="text-xs text-secondary">{conv.thread_id}</p>
                    </div>
                  </td>
                  <td className="px-4 py-3 text-secondary">
                    {formatNumber(conv.message_count)}
                  </td>
                  <td className="px-4 py-3 text-xs text-secondary">
                    {formatDateTime(conv.created_at)}
                  </td>
                  <td className="px-4 py-3 text-xs text-secondary">
                    {formatRelativeTime(conv.updated_at)}
                  </td>
                  <td className="px-4 py-3">
                    <button
                      onClick={(e) => handleDelete(e, conv.thread_id)}
                      className="rounded p-1 text-secondary hover:bg-red-50 hover:text-danger"
                      title="删除"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
