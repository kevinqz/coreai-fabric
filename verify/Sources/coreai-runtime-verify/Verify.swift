import Foundation
import CoreAI
import CoreAILanguageModels
import FoundationModels

// MARK: - Gate C: runtime loadability verifier
//
// Usage:
//   coreai-runtime-verify --model <bundle-dir-or-.aimodel> --kind llm  [--input "prompt"]
//   coreai-runtime-verify --model <bundle-dir-or-.aimodel> --kind graph
//
// Emits a JSON verdict on stdout. Exit code 0 = loads+runs, 1 = failed.
// The verdict records the OS build so the catalog can key a compatibility matrix
// (artifact × runtime → loads?), turning a producer *claim* into a consumer *guarantee*.

struct Verdict: Codable {
    let artifact: String
    let kind: String
    var loads: Bool = false
    var runs: Bool = false
    let runtime: Runtime
    var outputPreview: String? = nil
    var elapsedSeconds: Double? = nil
    var error: String? = nil
    let verifiedAt: String
    let verifierVersion = "coreai-runtime-verify/0.1"

    struct Runtime: Codable {
        let os: String            // e.g. "Version 27.0 (Build 26A5378j)"
        let arch: String
        let coreaiNote: String
    }
}

@main
struct Main {
    static func main() async {
        let args = parseArgs()
        let iso = ISO8601DateFormatter()
        let runtime = Verdict.Runtime(
            os: ProcessInfo.processInfo.operatingSystemVersionString,
            arch: machineArch(),
            coreaiNote: "system CoreAI framework; built with the SDK selected at build time"
        )
        var verdict = Verdict(artifact: args.model, kind: args.kind, runtime: runtime,
                              verifiedAt: iso.string(from: Date()))

        let start = Date()
        do {
            switch args.kind {
            case "llm":
                let preview = try await verifyLLM(dir: URL(fileURLWithPath: args.model),
                                                  prompt: args.input ?? "Reply with one short word.")
                verdict.loads = true; verdict.runs = true; verdict.outputPreview = preview
            case "graph":
                let shape = try await verifyGraph(bundle: URL(fileURLWithPath: args.model))
                verdict.loads = true; verdict.runs = true
                verdict.outputPreview = "output shape \(shape)"
            default:
                verdict.error = "unknown --kind \(args.kind) (use llm|graph)"
            }
        } catch {
            verdict.error = "\(error)"
        }
        verdict.elapsedSeconds = Date().timeIntervalSince(start)

        emit(verdict)
        exit(verdict.runs ? 0 : 1)
    }

    // MARK: verifiers

    static func verifyLLM(dir: URL, prompt: String) async throws -> String {
        let model = try await CoreAILanguageModel(resourcesAt: dir)
        let session = LanguageModelSession(model: model)
        let response = try await session.respond(to: prompt)
        return String(response.content.prefix(120))
    }

    /// Loads a graph and runs its first function with zero-filled inputs — proves
    /// the compiled IR loads AND executes a forward pass on this runtime.
    static func verifyGraph(bundle: URL) async throws -> [Int] {
        let url = aimodelURL(in: bundle)
        let model = try await AIModel(contentsOf: url,
                                      options: SpecializationOptions(preferredComputeUnitKind: .gpu))
        guard let fnName = model.functionNames.first,
              let fn = try model.loadFunction(named: fnName),
              let desc = model.functionDescriptor(for: fnName) else {
            throw Err.msg("no usable graph function")
        }
        var inputs: [String: NDArray] = [:]
        for name in desc.inputNames {
            if case .ndArray(let nd)? = desc.inputDescriptor(of: name) {
                inputs[name] = NDArray(descriptor: nd)   // zero-initialized
            }
        }
        var outputs = try await fn.run(inputs: inputs, outputViews: InferenceFunction.MutableViews())
        guard let outName = desc.outputNames.first,
              let out = outputs.remove(outName)?.ndArray else {
            throw Err.msg("function produced no output")
        }
        return out.shape
    }

    // MARK: helpers

    static func aimodelURL(in dir: URL) -> URL {
        if dir.pathExtension == "aimodel" { return dir }
        let items = (try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? []
        return items.first { $0.pathExtension == "aimodel" } ?? dir
    }

    static func machineArch() -> String {
        var s = utsname(); uname(&s)
        return withUnsafeBytes(of: &s.machine) { String(cString: $0.baseAddress!.assumingMemoryBound(to: CChar.self)) }
    }

    enum Err: Error, CustomStringConvertible { case msg(String); var description: String { if case .msg(let m) = self { return m }; return "error" } }

    struct Args { var model = ""; var kind = "llm"; var input: String? }
    static func parseArgs() -> Args {
        var a = Args(); var it = CommandLine.arguments.dropFirst().makeIterator()
        while let f = it.next() {
            switch f {
            case "--model": a.model = it.next() ?? ""
            case "--kind": a.kind = it.next() ?? "llm"
            case "--input": a.input = it.next()
            default: break
            }
        }
        return a
    }

    static func emit(_ v: Verdict) {
        let enc = JSONEncoder(); enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        if let data = try? enc.encode(v), let s = String(data: data, encoding: .utf8) { print(s) }
    }
}
