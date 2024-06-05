export interface File {
  id: string;
  name: string;
  size: number;
}

export interface Metadata {
  files?: File[];
  reference?: string;
}

export interface Message {
  role: 'user' | 'assistant' | 'system' | 'observation';
  metadata: string;
  content: string;
  request_metadata?: Metadata;
}

export interface ToolObservation {
  contentType: string;
  result: string;
  text?: string;
  roleMetadata?: string; // metadata for <|observation|>${metadata}
  metadata: any; // metadata for response
}
