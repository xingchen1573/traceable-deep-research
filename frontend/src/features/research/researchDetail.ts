import { ResearchBlueprintSection, ResearchJob, ResearchJobResult } from '@/types';

export type ResearchSectionRole = 'core' | 'supplemental';

export interface ResearchSectionMeta {
  sectionId: string;
  title: string;
  role: ResearchSectionRole;
  status?: string;
}

function cleanText(value: unknown): string {
  return typeof value === 'string' ? value.trim() : '';
}

export function deriveResearchDetailTitle(job: ResearchJob): string {
  const result = job.result;
  if (result?.pipeline_mode === 'single_url') {
    return cleanText(result.source_title)
      || cleanText(result.url)
      || cleanText(job.input?.url)
      || 'URL 简报';
  }
  return cleanText(job.input?.user_request)
    || cleanText(result?.report_blueprint?.title)
    || cleanText(result?.research_plan?.topic)
    || cleanText(result?.suggested_topic)
    || '调研任务';
}

function sectionKey(value: unknown): string {
  return cleanText(value).toLocaleLowerCase();
}

function definedFields<T extends object>(value?: T): Partial<T> {
  if (!value) return {};
  return Object.fromEntries(Object.entries(value).filter(([, field]) => field !== undefined)) as Partial<T>;
}

function sectionIdentities(section: ResearchBlueprintSection): string[] {
  return [...new Set([
    sectionKey(section.section_id),
    sectionKey(section.id),
    sectionKey(section.title),
  ].filter(Boolean))];
}

function mergeBlueprintSection(
  preferred: ResearchBlueprintSection,
  fallback: ResearchBlueprintSection,
): ResearchBlueprintSection {
  const reportability = {
    ...definedFields(fallback.reportability),
    ...definedFields(preferred.reportability),
  };
  const merged: ResearchBlueprintSection = {
    ...definedFields(fallback),
    ...definedFields(preferred),
    section_id: cleanText(preferred.section_id) || cleanText(fallback.section_id) || undefined,
    id: cleanText(preferred.id) || cleanText(fallback.id) || undefined,
    title: cleanText(preferred.title) || cleanText(fallback.title) || undefined,
  };
  if (Object.keys(reportability).length) merged.reportability = reportability;
  return merged;
}

function collectBlueprintSections(result?: ResearchJobResult | null): ResearchBlueprintSection[] {
  if (!result) return [];
  const sources: ResearchBlueprintSection[][] = [
    result.evidence_native_blueprint?.sections || [],
    result.report_blueprint?.sections || [],
    result.research_plan?.sections || [],
    (result.documents || []).map((document) => ({
      section_id: document.section_id,
      title: document.title,
      reportability: document.section_brief?.reportability,
    })),
  ];
  const merged: ResearchBlueprintSection[] = [];
  const indexByIdentity = new Map<string, number>();

  for (const sections of sources) {
    for (const section of sections) {
      const identities = sectionIdentities(section);
      const existingIndex = identities
        .map((identity) => indexByIdentity.get(identity))
        .find((index): index is number => index !== undefined);
      const index = existingIndex ?? merged.length;
      merged[index] = existingIndex === undefined
        ? mergeBlueprintSection(section, {})
        : mergeBlueprintSection(merged[index], section);
      for (const identity of [...identities, ...sectionIdentities(merged[index])]) {
        indexByIdentity.set(identity, index);
      }
    }
  }
  return merged;
}

export function deriveResearchSectionMetadata(result?: ResearchJobResult | null): ResearchSectionMeta[] {
  const byIdentity = new Map<string, ResearchSectionMeta>();
  for (const section of collectBlueprintSections(result)) {
    if (typeof section.reportability?.core_section !== 'boolean') continue;
    const sectionId = cleanText(section.section_id) || cleanText(section.id);
    const title = cleanText(section.title);
    if (!sectionId && !title) continue;
    const meta: ResearchSectionMeta = {
      sectionId,
      title,
      role: section.reportability.core_section ? 'core' : 'supplemental',
      status: cleanText(section.reportability.status) || undefined,
    };
    const identity = sectionKey(sectionId) || sectionKey(title);
    if (!byIdentity.has(identity)) byIdentity.set(identity, meta);
  }
  return [...byIdentity.values()];
}

export function findResearchSectionMeta(
  metadata: ResearchSectionMeta[],
  identity: unknown,
): ResearchSectionMeta | undefined {
  const key = sectionKey(identity);
  if (!key) return undefined;
  return metadata.find((section) => sectionKey(section.sectionId) === key || sectionKey(section.title) === key);
}
