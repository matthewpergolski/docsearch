import type { SearchResponse } from "../types";

export const RESPONSE: SearchResponse = {
  terms: ["freight", "region"],
  notes: [],
  results: {
    query: "Q3 freight cost by region",
    confident: true,
    answer: null,
    chunk_ids: ["a", "b", "c"],
    groups: [
      {
        rank: 1,
        file: "reports/supplier-deck-q3.pptx",
        format: "pptx",
        name: "Supplier review Q3",
        date: "2024-10-02",
        period: "Q3 2024",
        pages: [
          {
            page: 4,
            rank: 1,
            title: "Freight by region",
            snippet: "Freight cost per region rose 4% in the north.",
            score: 0.91,
            scorer: "rerank",
          },
          {
            page: 7,
            rank: 3,
            title: "",
            snippet: "Region totals for freight and handling.",
            score: 0.77,
            scorer: "rerank",
          },
          { page: 9, rank: 5, title: "", snippet: "Appendix.", score: 0.5, scorer: "rerank" },
          { page: 12, rank: 8, title: "", snippet: "Contacts.", score: 0.4, scorer: "rerank" },
        ],
      },
      {
        rank: 2,
        file: "finance/costs-2024.xlsx",
        format: "xlsx",
        name: "costs-2024.xlsx",
        date: "",
        period: "",
        pages: [
          {
            page: null,
            rank: 2,
            title: "Freight",
            snippet: "Region | Q3 freight | Q2 freight",
            score: 0.8,
            scorer: "rerank",
          },
        ],
      },
    ],
  },
};
