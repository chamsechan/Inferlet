#ifndef INFERLET_ENGINE_GUIDED_CHOICE_H_INCLUDED
#define INFERLET_ENGINE_GUIDED_CHOICE_H_INCLUDED

#include "support/status.h"
#include <string>
#include <vector>

namespace inferlet {

class GuidedChoice {
public:
    // Escape a string literal for GBNF grammar
    static std::string escape_gbnf_string(const std::string &str);

    // Build a GBNF grammar from a list of choices using Trie prefix merging.
    // Follows Section 6.8:
    // - JSON/GBNF string complete escaping
    // - exact choice followed only by EOG (handled naturally when root rule terminates)
    // - common prefix merged via Trie
    // - empty choices rejected
    static Status build_grammar(const std::vector<std::string> &choices, std::string &out_grammar);
};

} // namespace inferlet

#endif // INFERLET_ENGINE_GUIDED_CHOICE_H_INCLUDED
