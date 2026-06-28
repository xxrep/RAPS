"""
Legacy modules retained for reference only — NOT used by the RAPS coordinator.

- graph.py            : the GPTSwarm/G-Designer-style GCN graph optimizer this repo
                        was forked from (depends on torch_geometric and references
                        undefined GCN/MLP; non-functional).
- profile_embedding.py: local sentence-transformers embedding used only by graph.py.
                        The live broker now uses the Qwen3 embedding service.
- subscription.py     : an early SubscriptionTemplate superseded by
                        Node.refine_system_prompt (reactive subscription).
"""
