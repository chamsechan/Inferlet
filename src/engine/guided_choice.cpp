#include "guided_choice.h"
#include <map>
#include <memory>
#include <sstream>

namespace inferlet {

std::string GuidedChoice::escape_gbnf_string(const std::string &str) {
    std::string out;
    out.reserve(str.size() * 2);
    for (size_t i = 0; i < str.size(); ++i) {
        char c = str[i];
        if (c == '\\') {
            out += "\\\\";
        } else if (c == '"') {
            out += "\\\"";
        } else if (c == '\n') {
            out += "\\n";
        } else if (c == '\r') {
            out += "\\r";
        } else if (c == '\t') {
            out += "\\t";
        } else if (static_cast<unsigned char>(c) < 32) {
            char hex[8];
            snprintf(hex, sizeof(hex), "\\x%02x", static_cast<unsigned char>(c));
            out += hex;
        } else {
            out += c;
        }
    }
    return out;
}

namespace {

struct TrieNode {
    bool is_terminal = false;
    std::map<char, std::unique_ptr<TrieNode>> children;
};

struct RadixNode {
    bool is_terminal = false;
    std::map<std::string, std::unique_ptr<RadixNode>> children;
};

std::unique_ptr<RadixNode> compress(const TrieNode *trie) {
    auto radix = std::make_unique<RadixNode>();
    radix->is_terminal = trie->is_terminal;
    for (const auto &[ch, child_trie] : trie->children) {
        std::string edge(1, ch);
        const TrieNode *curr = child_trie.get();
        while (!curr->is_terminal && curr->children.size() == 1) {
            auto it = curr->children.begin();
            edge.push_back(it->first);
            curr = it->second.get();
        }
        radix->children[edge] = compress(curr);
    }
    return radix;
}

std::string serialize_node(const RadixNode *node) {
    if (node->children.empty()) {
        return "";
    }
    if (node->children.size() == 1) {
        const auto &[edge, child] = *node->children.begin();
        std::string sub = serialize_node(child.get());
        std::string edge_lit = "\"" + GuidedChoice::escape_gbnf_string(edge) + "\"";
        std::string res = sub.empty() ? edge_lit : (edge_lit + " " + sub);
        if (node->is_terminal) {
            return "(" + res + ")?";
        }
        return res;
    }

    std::vector<std::string> branches;
    for (const auto &[edge, child] : node->children) {
        std::string sub = serialize_node(child.get());
        std::string edge_lit = "\"" + GuidedChoice::escape_gbnf_string(edge) + "\"";
        branches.push_back(sub.empty() ? edge_lit : (edge_lit + " " + sub));
    }
    std::string inner;
    for (size_t i = 0; i < branches.size(); ++i) {
        if (i > 0) inner += " | ";
        inner += branches[i];
    }
    if (node->is_terminal) {
        return "((" + inner + "))?";
    } else {
        return "(" + inner + ")";
    }
}

} // namespace

Status GuidedChoice::build_grammar(const std::vector<std::string> &choices, std::string &out_grammar) {
    if (choices.empty()) {
        return Status::invalid_argument("Choices list cannot be empty for guided choices");
    }

    TrieNode root;
    for (const auto &choice : choices) {
        if (choice.empty()) {
            return Status::invalid_argument("Empty string choice is not allowed in guided choices");
        }
        TrieNode *curr = &root;
        for (char c : choice) {
            auto &child = curr->children[c];
            if (!child) {
                child = std::make_unique<TrieNode>();
            }
            curr = child.get();
        }
        curr->is_terminal = true;
    }

    auto radix = compress(&root);
    if (radix->children.empty()) {
        return Status::invalid_argument("Failed to build Trie for guided choices");
    }

    if (radix->children.size() == 1) {
        const auto &[edge, child] = *radix->children.begin();
        std::string sub = serialize_node(child.get());
        std::string edge_lit = "\"" + escape_gbnf_string(edge) + "\"";
        std::string body = sub.empty() ? edge_lit : (edge_lit + " " + sub);
        out_grammar = "root ::= " + body + "\n";
    } else {
        std::vector<std::string> branches;
        for (const auto &[edge, child] : radix->children) {
            std::string sub = serialize_node(child.get());
            std::string edge_lit = "\"" + escape_gbnf_string(edge) + "\"";
            branches.push_back(sub.empty() ? edge_lit : (edge_lit + " " + sub));
        }
        std::string inner;
        for (size_t i = 0; i < branches.size(); ++i) {
            if (i > 0) inner += " | ";
            inner += branches[i];
        }
        out_grammar = "root ::= (" + inner + ")\n";
    }

    return Status::ok();
}

} // namespace inferlet
