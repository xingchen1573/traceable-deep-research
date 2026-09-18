import { adapter } from './adapters';
import { systemEndpoints } from './endpoints/system';
import { knowledgeBaseEndpoints } from './endpoints/knowledgeBases';
import { researchEndpoints } from './endpoints/research';
import { evaluationEndpoints } from './endpoints/evaluations';
import { observabilityEndpoints } from './endpoints/observability';
import { modelEndpoints } from './endpoints/models';
import { jobEndpoints } from './endpoints/jobs';
import { chatEndpoints } from './endpoints/chat';
import { channelEndpoints } from './endpoints/channels';
import { conversationEndpoints } from './endpoints/conversations';
import { knowledgeTreeEndpoints } from './endpoints/knowledgeTree';

export const api = {
  system: systemEndpoints(adapter),
  knowledgeBases: knowledgeBaseEndpoints(adapter),
  research: researchEndpoints(adapter),
  evaluations: evaluationEndpoints(adapter),
  observability: observabilityEndpoints(adapter),
  models: modelEndpoints(adapter),
  jobs: jobEndpoints(adapter),
  chat: chatEndpoints(adapter),
  channels: channelEndpoints(adapter),
  conversations: conversationEndpoints(adapter),
  knowledgeTree: knowledgeTreeEndpoints(adapter),
};

export type Api = typeof api;
