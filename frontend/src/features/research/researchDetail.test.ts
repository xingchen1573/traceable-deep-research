import { describe, expect, it } from 'vitest';
import { ResearchJob } from '@/types';
import { deriveResearchDetailTitle, deriveResearchSectionMetadata, findResearchSectionMeta } from './researchDetail';

function makeJob(overrides: Partial<ResearchJob> = {}): ResearchJob {
  return {
    job_id: 'job_fixture_research_detail_001',
    job_type: 'research',
    status: 'waiting_approval',
    created_at: '2026-01-01 10:00:00',
    updated_at: '2026-01-01 10:05:00',
    input: {},
    result: null,
    error: null,
    artifacts: [],
    ...overrides,
  };
}

describe('deriveResearchDetailTitle', () => {
  it('prefers the readable original request over a filesystem-safe topic slug', () => {
    const job = makeJob({
      input: { user_request: '调研并比较 AlphaResearch、BetaSearch 和 GammaLab' },
      result: { status: 'waiting_approval', suggested_topic: '调研并比较_alphasearch_betasearch_和_gammalab' },
    });
    expect(deriveResearchDetailTitle(job)).toBe('调研并比较 AlphaResearch、BetaSearch 和 GammaLab');
  });

  it('uses the source title for a single URL brief', () => {
    const job = makeJob({
      input: { user_request: '总结一下', url: 'https://example.com/article' },
      result: { status: 'succeeded', pipeline_mode: 'single_url', source_title: '示例文章' },
    });
    expect(deriveResearchDetailTitle(job)).toBe('示例文章');
  });
});

describe('deriveResearchSectionMetadata', () => {
  it('projects core/supplemental roles and reportability status from the evidence-native blueprint', () => {
    const result = {
      status: 'waiting_approval',
      evidence_native_blueprint: {
        sections: [
          { section_id: 'sec_01', title: '01_核心机制', reportability: { core_section: true, status: 'empty' } },
          { section_id: 'sec_02', title: '02_补充案例', reportability: { core_section: false, status: 'ready' } },
        ],
      },
    };
    const metadata = deriveResearchSectionMetadata(result);
    expect(findResearchSectionMeta(metadata, '01_核心机制')).toMatchObject({ role: 'core', status: 'empty' });
    expect(findResearchSectionMeta(metadata, 'sec_02')).toMatchObject({ role: 'supplemental', status: 'ready' });
  });

  it('falls back to a document section brief for compatible historical jobs', () => {
    const metadata = deriveResearchSectionMetadata({
      status: 'waiting_approval',
      documents: [{
        section_id: 'sec_03',
        title: '03_报告生成',
        content: '',
        section_brief: { reportability: { core_section: true, status: 'limited' } },
      }],
    });
    expect(metadata).toEqual([{ sectionId: 'sec_03', title: '03_报告生成', role: 'core', status: 'limited' }]);
  });

  it('fills missing primary reportability fields from a matching historical document section', () => {
    const metadata = deriveResearchSectionMetadata({
      status: 'waiting_approval',
      evidence_native_blueprint: {
        sections: [{
          section_id: 'sec_legacy',
          title: '04_历史章节',
          reportability: { status: 'limited' },
        }],
      },
      documents: [{
        section_id: 'sec_legacy',
        title: '04_历史章节',
        content: '',
        section_brief: { reportability: { core_section: true, status: 'ready', evidence_count: 2 } },
      }],
    });

    expect(metadata).toEqual([{
      sectionId: 'sec_legacy',
      title: '04_历史章节',
      role: 'core',
      status: 'limited',
    }]);
  });

  it('uses a matching document brief when a historical primary section has no reportability object', () => {
    const metadata = deriveResearchSectionMetadata({
      status: 'waiting_approval',
      report_blueprint: {
        sections: [{ section_id: 'sec_without_reportability', title: '05_历史补充' }],
      },
      documents: [{
        section_id: 'sec_without_reportability',
        title: '05_历史补充',
        content: '',
        section_brief: { reportability: { core_section: false, status: 'ready' } },
      }],
    });

    expect(metadata).toEqual([{
      sectionId: 'sec_without_reportability',
      title: '05_历史补充',
      role: 'supplemental',
      status: 'ready',
    }]);
  });
});
