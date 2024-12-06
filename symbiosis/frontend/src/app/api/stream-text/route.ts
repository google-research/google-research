import {createGoogleGenerativeAI} from '@ai-sdk/google';
import { streamText, convertToCoreMessages } from 'ai';

// Allow streaming responses up to 30 seconds
export const maxDuration = 30;

export async function POST(req: Request) {
  const { messages } = await req.json();

  const google = createGoogleGenerativeAI({
    apiKey: process.env.GEMINI_API_KEY,
  })

  const result = await streamText({
    model: google('gemini-1.5-flash-latest'),
    messages: convertToCoreMessages(messages),
  });

  return result.toDataStreamResponse();
}