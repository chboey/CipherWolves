import JSZip from 'jszip';
import { Conversation } from '../post-game/page';
import { AgentMetrics } from '@/components/metricsChart';

export const useDataDownload = () => {
  const downloadData = async (
    gameKeywords?: Array<string>,
    conversation?: Conversation[],
    metrics?: AgentMetrics[],
    filename?: string
  ) => {
    try {
      const zip = new JSZip();

      zip.file('gamekeywords.json', JSON.stringify(gameKeywords, null, 2));
      zip.file('conversation.json', JSON.stringify(conversation, null, 2));
      zip.file('trust-metrics.json', JSON.stringify(metrics, null, 2));

      const zipBlob = await zip.generateAsync({ type: 'blob' });

      const url = URL.createObjectURL(zipBlob);
      const link = document.createElement('a');
      link.href = url;
      link.download =
        `${filename}-${Date.now()}.zip` || `data-export-${Date.now()}.zip`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Error downloading data:', error);
      throw error;
    }
  };

  return { downloadData };
};
