import {createGoogleGenerativeAI} from '@ai-sdk/google';
import { generateObject } from 'ai';
import { z } from 'zod';

const schema = z.object({
  variables: z.array(
      z.object({
        id: z.string().describe('Unique identifier for the node'),
        type: z.string().trim().describe('Type of the node. Always use "variablenode" value.'),
        position: z.object({
          x: z.number().describe('X coordinate of the node.'),
          y: z.number().describe('Y coordinate of the node.'),
        }),
        data: z.object({
          label: z.string().describe('Label to be displayed on the node'),
        }),
      }),
  ).describe('Array of variables in the causal loop diagram'),
  causallinks: z.array(
      z.object({
        id: z.string().describe('Unique identifier for the causal link in the causal loop diagram between the variables'),
        source: z.string().describe('ID of the source variable'),
        target: z.string().describe('ID of the target variable'),
        data: z.object({
          endLabel: z.string().trim().describe('Polarity of the causal link in the causal loop diagram. Only use + or - values for positive and negative polarity respectively.'),
        }),
        type: z.string().trim().describe('Type of the causal link. Always use "customcircular" value.'),
        markerEnd: z.object({
          type: z.string().describe('Type of marker at the end of the causal link. Always use "MarkerType.ArrowClosed" value.'),
          color: z.string().trim().describe('Color of the marker. Always use "#FF0072" value.'),
        }),
        style: z.object({
          strokeWidth: z.number().describe('Width of the causal link stroke. Always use 2 value.'),
          stroke: z.string().trim().describe('Color of the causal link stroke. Always use "#FF0072" value.'),
        }),
        animated: z.boolean().optional().describe('Whether the causal link is animated. Always use "true" value.'),
      }),
  ).describe('Array of causal links connecting the variables'),
});

export async function POST(req: Request) {
  let { prompt }: { prompt: string } = await req.json();
  const google = createGoogleGenerativeAI({
    apiKey: process.env.GEMINI_API_KEY,
  })

  const result = await generateObject({
    model: google('gemini-1.5-flash-latest'),
    system: `You are a System Dynamics expert tasked with creating causal loop diagrams (CLDs) from user-provided text.  Your goal is to generate a holistic CLD that accurately reflects the causal relationships described in the text, incorporating feedback loops and auxiliary variables where applicable.
    If the user provided text is simply a reference node or topic, then use your own knowledge to create the causal loop diagram and do you own research on relevant variables and causal links. Use the topic or keyword or phrase as the reference node and make sure it's represented in the variables.
    DO NOT limit yourself to only one feedback loop. Suggest atleast 10 variables, 10 causal links, 3 feedback loops, many auxiliary variables where applicable. Be as complete as possible.
    Here's your process:

      - Identify Variables: Extract key concepts from the text that have cause-and-effect relationships. Represent these as concise, neutral variable names (max 2 words). Minimize the number of variables used.

      - Establish Causal Links: Determine the polarity of the causal relationships between variables.

      - Positive Polarity (+): An increase in one variable leads to an increase in another, or a decrease in one leads to a decrease in the other. (Example: "Motivation" -->(+) "Productivity")
      - Negative Polarity (-): An increase in one variable leads to a decrease in another, and vice versa. (Example: "Stress" -->(-) "Job Satisfaction")
      - Test Polarity: To confirm polarity, ask "If this variable increases, what happens to the other variable?"
      - Form Feedback Loops: Connect causal links to form closed loops.

      - Reinforcing Loop (R): An even number of negative links, or all positive links, indicating a self-amplifying effect. (Example: Increased sales lead to higher profits, which allow for more investment, further boosting sales.)
      - Balancing Loop (B): An odd number of negative links, indicating a self-correcting effect. (Example: Increased prices lead to lower demand, eventually reducing prices.)
      - Incorporate Auxiliary Variables: When necessary, introduce auxiliary variables to clarify relationships or provide additional context. These variables should not be the main focus but help explain the dynamics between primary variables.
`,
    prompt,
    schemaName: 'CreateCausalLoopDiagram',
    schemaDescription: 'Variables and Causal Links of the causal loop diagram',
    schema: schema,
  });

  return result.toJsonResponse();
}