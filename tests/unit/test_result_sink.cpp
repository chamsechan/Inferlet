#include "core/stream_sink.h"
#include <cassert>
#include <iostream>

using namespace inferlet;

static TokenStep token_piece(std::string piece) {
    TokenStep step;
    step.token = 42;
    step.piece = std::move(piece);
    return step;
}

static void test_stream_finish_errors() {
    for (const auto &error : {Status::inference_error("decode failed"), Status::cancelled("cancelled")}) {
        auto core = std::make_shared<TaskCore>();
        StreamingSink sink(core);
        assert(sink.on_token(token_piece("")).is_ok());
        sink.on_finish(error);
        assert(sink.finish_status().code == error.code);
        assert(sink.finish_status().message == error.message);
        assert(core->chunks.empty());
    }

    auto core = std::make_shared<TaskCore>();
    StreamingSink sink(core);
    assert(sink.on_token(token_piece("")).is_ok());
    core->cancel_requested.store(true);
    sink.on_finish(Status::ok());
    assert(sink.finish_status().code == INFERLET_E_CANCELLED);
    assert(core->chunks.empty());
}

static void test_incomplete_utf8() {
    auto core = std::make_shared<TaskCore>();
    StreamingSink stream(core);
    assert(stream.on_token(token_piece("\xF4")).is_ok());
    stream.on_finish(Status::ok());
    assert(stream.finish_status().code == INFERLET_E_INFERENCE);
    assert(core->chunks.empty());

    for (const auto &bytes : {std::string("\xF4"), std::string("\xFF")}) {
        TaskOutputInternal output;
        CollectingSink sink(output);
        assert(sink.on_token(token_piece(bytes)).is_ok());
        sink.on_finish(Status::ok());
        assert(sink.finish_status().code == INFERLET_E_INFERENCE);
    }
}

static void test_complete_utf8_and_error_precedence() {
    TaskOutputInternal output;
    CollectingSink sink(output);
    assert(sink.on_token(token_piece("\xE4")).is_ok());
    assert(sink.on_token(token_piece("\xBD\xA0")).is_ok());
    sink.on_finish(Status::ok());
    assert(sink.finish_status().is_ok());
    assert(output.result.text == u8"你");
    assert(output.result.tokens.size() == 2);

    TaskOutputInternal partial;
    CollectingSink cancelled(partial);
    assert(cancelled.on_token(token_piece("\xE4")).is_ok());
    cancelled.on_finish(Status::cancelled("original error"));
    assert(cancelled.finish_status().code == INFERLET_E_CANCELLED);
    assert(cancelled.finish_status().message == "original error");
}

int main() {
    test_stream_finish_errors();
    test_incomplete_utf8();
    test_complete_utf8_and_error_precedence();
    std::cout << "Result sink error and UTF-8 tests passed.\n";
}
