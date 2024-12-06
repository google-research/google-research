import {createGoogleGenerativeAI} from '@ai-sdk/google';
import { generateObject } from 'ai';
import { z } from 'zod';

const schema = z.object({
  suggestions: z.array(
      z.string().describe('Suggestion (Maximum 5 words.)')
  )
});

export async function POST(req: Request) {
  let { prompt }: { prompt: string } = await req.json();
  const google = createGoogleGenerativeAI({
    apiKey: process.env.GEMINI_API_KEY,
  })

  const result = await generateObject({
    model: google('gemini-1.5-flash-latest'),
    system: 'You are a helpful assistant and you are going to help me create prompt suggestions that I should provide to the user based on the content in the prompt.' +
        'The goal is to enable user to explore the causal loop diagram. DO NOT PROVIDE SUGGESTIONS RELATED TO MODIFYING THE CAUSAL LOOP DIAGRAM.' +
        'PROVIDE MINIMUM 5 AND MAXIMUM 10 SUGGESTIONS. SUGGESTIONS SHOULD ONLY BE RELATED TO EXPLORING OR UNDERSTANDING THE CAUSAL LOOP DIAGRAM.' +
        'SOME SUGGESTIONS SHOULD BE SPECIFIC TO THE CAUSAL LOOP DIAGRAM IN QUESTIONS (FOR EXAMPLE YOU CAN PROVIDE, Explain relationship between X and Y variables, where X and Y are the variables in the causal loop diagram.',
    prompt,
    schemaName: 'PromptSuggestions',
    schemaDescription: 'Prompt Suggestions based on the content of the causal loop diagram',
    schema: schema,
  });

  return result.toJsonResponse();
}