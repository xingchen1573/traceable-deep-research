import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, User, Bot, Trash2 } from 'lucide-react';
import { PageHeader } from '@/components/layout/PageHeader';
import { ErrorPanel } from '@/components/data/ErrorPanel';
import { Button } from '@/components/controls/Button';
import { useConversationMessages, useDeleteConversation } from '@/hooks';
import { formatDateTime } from '@/utils/formatters';
import type { ConversationMessage } from '@/api/endpoints/conversations';

function MessageBubble({ msg }: { msg: ConversationMessage }) {
  const isUser = msg.role === 'user';
  return (
    <div className={`flex gap-3 ${isUser ? 'flex-row-reverse' : ''}`}>
      <div
        className={`flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full ${
          isUser ? 'bg-accent/10 text-accent' : 'bg-page text-secondary'
        }`}
      >
        {isUser ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
      </div>
      <div
        className={`max-w-[80%] rounded-lg px-4 py-2.5 text-sm leading-relaxed ${
          isUser
            ? 'bg-accent text-white'
            : 'bg-page text-primary'
        }`}
      >
        <p className="whitespace-pre-wrap break-words">{msg.content}</p>
        <p className={`mt-1 text-xs ${isUser ? 'text-white/60' : 'text-secondary'}`}>
          {formatDateTime(msg.created_at)}
        </p>
      </div>
    </div>
  );
}

export function ConversationDetailPage() {
  const { threadId } = useParams<{ threadId: string }>();
  const navigate = useNavigate();
  const decodedThreadId = threadId ? decodeURIComponent(threadId) : '';
  const { data, isLoading, error, refetch } = useConversationMessages(decodedThreadId);
  const deleteConv = useDeleteConversation();

  const handleDelete = () => {
    if (confirm('确定删除该对话记录？此操作不可恢复。')) {
      deleteConv.mutate(decodedThreadId, {
        onSuccess: () => navigate('/conversations'),
      });
    }
  };

  return (
    <div className="space-y-4">
      <PageHeader
        title={data?.conversation?.title || decodedThreadId || '对话详情'}
        description={
          data?.conversation
            ? `${formatDateTime(data.conversation.created_at)} · ${data.messages.length} 条消息`
            : undefined
        }
        actions={
          <div className="flex gap-2">
            <Button variant="secondary" onClick={() => navigate('/conversations')} icon={undefined}>
              <ArrowLeft className="mr-1.5 h-4 w-4" />
              返回列表
            </Button>
            <Button variant="secondary" onClick={handleDelete} icon={undefined}>
              <Trash2 className="mr-1.5 h-4 w-4" />
              删除
            </Button>
          </div>
        }
      />

      {error && <ErrorPanel error={error} onRetry={() => refetch()} />}

      {isLoading ? (
        <div className="rounded-panel border border-border bg-white p-8 text-center text-secondary">
          加载中...
        </div>
      ) : !data?.messages?.length ? (
        <div className="rounded-panel border border-border bg-white p-8 text-center text-secondary">
          暂无消息记录
        </div>
      ) : (
        <div className="rounded-panel border border-border bg-white p-6">
          <div className="space-y-4">
            {data.messages.map((msg) => (
              <MessageBubble key={msg.id} msg={msg} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
