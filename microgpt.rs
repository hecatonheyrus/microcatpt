
// Rust port of @karpathy's microgpt.py
// https://karpathy.github.io/2026/02/12/microgpt/

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::rc::Rc;
use std::cell::RefCell;

// Random number generation
use rand::prelude::*;
use rand::distributions::WeightedIndex;


#[derive(Clone)]
struct Value {
    data: f64,
    grad: RefCell<f64>,
    children: Vec<Rc<Value>>,
    local_grads: Vec<f64>,
}

impl Value {

    fn new(data: f64, children: Vec<Rc<Value>>, local_grads: Vec<f64>) -> Rc<Self> {
        Rc::new(Value {
            data,
            grad: RefCell::new(0.0),
            children,
            local_grads,
        })
    }


    fn add(a: &Rc<Value>, b: &Rc<Value>) -> Rc<Value> {
     
        Value::new(
            a.data + b.data,
            vec![Rc::clone(a), Rc::clone(b)],
            vec![1.0, 1.0],
        )
    
    }

    fn mul(a: &Rc<Value>, b: &Rc<Value>) -> Rc<Value> {
        Value::new(
            a.data * b.data,
            vec![Rc::clone(a), Rc::clone(b)],
            vec![b.data, a.data],
        )
    }

    fn pow(a: &Rc<Value>, exp: f64) -> Rc<Value> {
        Value::new(
            a.data.powf(exp),
            vec![Rc::clone(a)],
            vec![exp * a.data.powf(exp - 1.0)],
        )
    }

    fn log(a: &Rc<Value>) -> Rc<Value> {
        Value::new(
            a.data.ln(),
            vec![Rc::clone(a)],
            vec![1.0 / a.data],
        )
    }

    fn exp(a: &Rc<Value>) -> Rc<Value> {
        let exp_val = a.data.exp();
        Value::new(exp_val, vec![Rc::clone(a)], vec![exp_val])
    }

    fn relu(a: &Rc<Value>) -> Rc<Value> {
        Value::new(
            a.data.max(0.0),
            vec![Rc::clone(a)],
            vec![if a.data > 0.0 { 1.0 } else { 0.0 }],
        )
    }

    
    fn neg(a: &Rc<Value>) -> Rc<Value> {
 
       Value::new(
            a.data * -1.0,
            vec![Rc::clone(a)],
            vec![a.data, -1.0],
        )

    }
    

    fn sub(a: &Rc<Value>, b: &Rc<Value>) -> Rc<Value> {
         Value::new(
            a.data - b.data,
            vec![Rc::clone(a), Rc::clone(b)],
            vec![1.0, 1.0],
        )
    }

    fn div(a: &Rc<Value>, b: &Rc<Value>) -> Rc<Value> {
        Value::mul(a, &Value::pow(b, -1.0))
    }

     fn build_topo(v: &Rc<Value>, visited: &mut HashSet<*const Value>, topo: &mut Vec<Rc<Value>>) {
            let ptr = Rc::as_ptr(v);
            if !visited.contains(&ptr) {
                visited.insert(ptr);
                for child in &v.children {
                    Self::build_topo(child, visited, topo);
                }
                topo.push(Rc::clone(v));
            }
        }

    fn backward(self: &Rc<Self>) {
        let mut topo = Vec::new();
        let mut visited = HashSet::new();
   

        Self::build_topo(self, &mut visited, &mut topo);
        *self.grad.borrow_mut() = 1.0;

        for v in topo.iter().rev() {
            let v_grad = *v.grad.borrow();
            for (child, &local_grad) in v.children.iter().zip(v.local_grads.iter()) {
                *child.grad.borrow_mut() += local_grad * v_grad;
            }
        }
    }

    fn zero_grad(&self) {
        *self.grad.borrow_mut() = 0.0;
    }
}

// ============================================================================
// Helper functions
// ============================================================================

fn has_double_characters(line: String) -> bool {

    let mut counts = std::collections::HashMap::new();

    for char in line.chars() {
        *counts.entry(char).or_insert(0) += 1;

        if *counts.get(&char).unwrap() > 1 {
            return true;
        }
    }

    false
}

fn replace_a_with_o(input: String) -> String {
    let mut result = String::new();

    let mut i = 0;
    for char in input.chars().rev() {
        if char == 'a' && i == 0{
            result.push('o');
        } else {
            result.push(char);
        }

        i += 1;
    }

    result
}

fn download_dataset() -> std::io::Result<()> {
    if !Path::new("input.txt").exists() {
        println!("Downloading dataset...");
        let url = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt";
        let response = ureq::get(url).call().map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
        })?;
        let mut file = File::create("input.txt")?;
        std::io::copy(&mut response.into_reader(), &mut file)?;
    }
    Ok(())
}

fn catify_it(line: String) -> String{

   let consonants = vec![Some('a'), Some('o'), Some('u'), Some('e'), Some('i'), Some('y')];

   let last_letter = line.chars().nth(line.len() - 1);

   for con in consonants{
       if con == last_letter{
          return line;
       }
   }

   let mut catified: String = line.to_owned();

   //catified.push(last_letter.unwrap());   // doubling
   match last_letter{
     Some('l') | Some('m') | Some('n') | Some('k') | Some('t') => {catified.push('i');
                                                                  catified.push('e');},
     _ => catified.push('y')                                                             
 }

   catified

}

fn load_docs() -> Vec<String> {
    let file = File::open("input.txt").expect("Failed to open input.txt");
    let reader = BufReader::new(file);
    reader
        .lines()
        .filter_map(|line| line.ok())
        .filter(|line| !line.trim().is_empty())
        .map(|line| catify_it(line))
        //.map(|line| replace_a_with_o(line))
        .filter(|line| line.chars().nth(line.len() - 1) == Some('i') || line.chars().nth(line.len() - 1) == Some('y')
                       || (line.chars().nth(line.len() - 1) == Some('e') && line.chars().nth(line.len() - 2) == Some('i'))
                       || (line.chars().nth(line.len() - 1) == Some('r') && line.chars().nth(line.len() - 2) == Some('e'))
                       || (line.chars().nth(line.len() - 1) == Some('o'))
               //        || has_double_characters(line.to_string())
        )     // kitties            
        .map(|line| line.trim().to_string())
        .collect()
}

// ============================================================================
// Model functions
// ============================================================================

fn linear(x: &[Rc<Value>], w: &[Vec<Rc<Value>>]) -> Vec<Rc<Value>> {
    w.iter()
        .map(|wo| {
            wo.iter()
                .zip(x.iter())
                .fold(Value::new(0.0, Vec::new(), Vec::new()), |acc, (wi, xi)| Value::add(&acc, &Value::mul(wi, xi)))
        })
        .collect()
}

fn softmax(logits: &[Rc<Value>]) -> Vec<Rc<Value>> {
    let max_val = logits.iter().map(|v| v.data).fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<Rc<Value>> = logits
        .iter()
        .map(|val| Value::exp(&Value::sub(val, &Value::new(max_val, Vec::new(), Vec::new()))))
        .collect();
    let total = exps.iter().fold(Value::new(0.0, Vec::new(), Vec::new()), |acc, e| Value::add(&acc, e));
    exps.iter().map(|e| Value::div(e, &total)).collect()
}

fn rmsnorm(x: &[Rc<Value>]) -> Vec<Rc<Value>> {
    let ms = x
        .iter()
        .fold(Value::new(0.0, Vec::new(), Vec::new()), |acc, xi| Value::add(&acc, &Value::mul(xi, xi)));
    let ms = Value::div(&ms, &Value::new(x.len() as f64, Vec::new(), Vec::new()));
    let scale = Value::pow(&Value::add(&ms, &Value::new(1e-5, Vec::new(), Vec::new())), -0.5);
    x.iter().map(|xi| Value::mul(xi, &scale)).collect()
}

fn attention(state_dict: &HashMap<String, Vec<Vec<Rc<Value>>>>,
             keys: &mut [Vec<Vec<Rc<Value>>>],
             values: &mut [Vec<Vec<Rc<Value>>>],
             n_head: usize,
             head_dim: usize,
             li: usize,
             x: &mut Vec<Rc<Value>>,
            ) -> Vec<Rc<Value>>{

            let mut x_attention = Vec::<Rc<Value>>::new();

            let q = linear(&x, &state_dict[&format!("layer{}.attn_wq", li)]);

        for h in 0..n_head {
            let hs = h * head_dim;
            let q_h = &q[hs..hs + head_dim];
            let k_h: Vec<Vec<Rc<Value>>> = keys[li]
                .iter()
                .map(|ki| ki[hs..hs + head_dim].to_vec())
                .collect();
            let v_h: Vec<Vec<Rc<Value>>> = values[li]
                .iter()
                .map(|vi| vi[hs..hs + head_dim].to_vec())
                .collect();

            let attn_logits: Vec<Rc<Value>> = (0..k_h.len())
                .map(|t| {
                    let sum = (0..head_dim).fold(Value::new(0.0, Vec::new(), Vec::new()), |acc, j| {
                        Value::add(&acc, &Value::mul(&q_h[j], &k_h[t][j]))
                    });
                    Value::div(&sum, &Value::new((head_dim as f64).sqrt(), Vec::new(), Vec::new()))
                })
                .collect();

            let attn_weights = softmax(&attn_logits);
            let head_out: Vec<Rc<Value>> = (0..head_dim)
                .map(|j| {
                    (0..v_h.len()).fold(Value::new(0.0, Vec::new(), Vec::new()), |acc, t| {
                        Value::add(&acc, &Value::mul(&attn_weights[t], &v_h[t][j]))
                    })
                })
                .collect();
            x_attention.extend(head_out);
        }



           x_attention

}

fn gpt(
    token_id: usize,
    pos_id: usize,
    keys: &mut [Vec<Vec<Rc<Value>>>],
    values: &mut [Vec<Vec<Rc<Value>>>],
    state_dict: &HashMap<String, Vec<Vec<Rc<Value>>>>,
    n_layer: usize,
    n_head: usize,
    head_dim: usize,
) -> Vec<Rc<Value>> {
    let tok_emb = &state_dict["wte"][token_id];
    let pos_emb = &state_dict["wpe"][pos_id];
    let mut x: Vec<Rc<Value>> = tok_emb
        .iter()
        .zip(pos_emb.iter())
        .map(|(t, p)| Value::add(t, p))
        .collect();
    x = rmsnorm(&x);

    for li in 0..n_layer {
        // Multi-head Attention block
        let x_residual = x.clone();
        x = rmsnorm(&x);
        let q = linear(&x, &state_dict[&format!("layer{}.attn_wq", li)]);
        let k = linear(&x, &state_dict[&format!("layer{}.attn_wk", li)]);
        let v = linear(&x, &state_dict[&format!("layer{}.attn_wv", li)]);
        keys[li].push(k.clone());
        values[li].push(v.clone());

        let mut x_attn = attention(state_dict, keys, values, n_head, head_dim, li, &mut x);
       

        x = linear(&x_attn, &state_dict[&format!("layer{}.attn_wo", li)]);
        x = x
            .iter()
            .zip(x_residual.iter())
            .map(|(a, b)| Value::add(a, b))
            .collect();

        // MLP block
        let x_residual = x.clone();
        x = rmsnorm(&x);
        x = linear(&x, &state_dict[&format!("layer{}.mlp_fc1", li)]);
        x = x.iter().map(|xi| Value::relu(xi)).collect();
        x = linear(&x, &state_dict[&format!("layer{}.mlp_fc2", li)]);
        x = x
            .iter()
            .zip(x_residual.iter())
            .map(|(a, b)| Value::add(a, b))
            .collect();
    }

    linear(&x, &state_dict["lm_head"])
}

// ============================================================================
// Main
// ============================================================================

fn main() -> std::io::Result<()> {
    let mut rng = StdRng::seed_from_u64(42);

    // Load dataset
    download_dataset()?;
    let mut docs = load_docs();

    docs.push("fluffy".to_string());
    docs.push("muffy".to_string());
    docs.push("duffy".to_string());
    docs.push("tootsie".to_string());
    

    docs.shuffle(&mut rng);
    println!("num docs: {}", docs.len());

    // Build vocabulary
    let mut uchars: Vec<char> = docs.iter().flat_map(|s| s.chars()).collect();
    uchars.sort_unstable();
    uchars.dedup();
    let bos = uchars.len();
    let vocab_size = uchars.len() + 1;
    println!("vocab size: {}", vocab_size);

    // Model hyperparameters
    let n_layer = 1;
    let n_embd = 16;
    let block_size = 16;
    let n_head = 4;
    let head_dim = n_embd / n_head;

    // Initialize parameters
    let matrix = |nout: usize, nin: usize, std: f64, rng: &mut StdRng| -> Vec<Vec<Rc<Value>>> {
        (0..nout)
            .map(|_| {
                (0..nin)
                    .map(|_| Value::new(rng.sample::<f64, _>(rand_distr::StandardNormal) * std, Vec::new(), Vec::new()))
                    .collect()
            })
            .collect()
    };

    let mut state_dict: HashMap<String, Vec<Vec<Rc<Value>>>> = HashMap::new();
    state_dict.insert("wte".to_string(), matrix(vocab_size, n_embd, 0.08, &mut rng));
    state_dict.insert("wpe".to_string(), matrix(block_size, n_embd, 0.08, &mut rng));
    state_dict.insert("lm_head".to_string(), matrix(vocab_size, n_embd, 0.08, &mut rng));

    for i in 0..n_layer {
        state_dict.insert(format!("layer{}.attn_wq", i), matrix(n_embd, n_embd, 0.08, &mut rng));
        state_dict.insert(format!("layer{}.attn_wk", i), matrix(n_embd, n_embd, 0.08, &mut rng));
        state_dict.insert(format!("layer{}.attn_wv", i), matrix(n_embd, n_embd, 0.08, &mut rng));
        state_dict.insert(format!("layer{}.attn_wo", i), matrix(n_embd, n_embd, 0.08, &mut rng));
        state_dict.insert(format!("layer{}.mlp_fc1", i), matrix(4 * n_embd, n_embd, 0.08, &mut rng));
        state_dict.insert(format!("layer{}.mlp_fc2", i), matrix(n_embd, 4 * n_embd, 0.08, &mut rng));
    }

    let params: Vec<Rc<Value>> = state_dict
        .values()
        .flat_map(|mat| mat.iter().flat_map(|row| row.iter().cloned()))
        .collect();
    println!("num params: {}", params.len());

    // Adam optimizer buffers
    let learning_rate = 0.01;
    let beta1 = 0.85;
    let beta2 = 0.99;
    let eps_adam = 1e-8;
    let mut m = vec![0.0; params.len()];
    let mut v = vec![0.0; params.len()];

    // Training loop
    let num_steps = 1000;
    for step in 0..num_steps {
        let doc = &docs[step % docs.len()];
        let mut tokens = vec![bos];
        tokens.extend(doc.chars().map(|ch| uchars.iter().position(|&c| c == ch).unwrap()));
        tokens.push(bos);
        let n = tokens.len().min(block_size + 1) - 1;

        // Forward pass
        let mut keys = vec![Vec::new(); n_layer];
        let mut values = vec![Vec::new(); n_layer];
        let mut losses = Vec::new();

        for pos_id in 0..n {
            let token_id = tokens[pos_id];
            let target_id = tokens[pos_id + 1];
            let logits = gpt(
                token_id,
                pos_id,
                &mut keys,
                &mut values,
                &state_dict,
                n_layer,
                n_head,
                head_dim,
            );
            let probs = softmax(&logits);
            let loss_t = Value::neg(&Value::log(&probs[target_id]));
            losses.push(loss_t);
        }

        let loss = Value::div(
            &losses.iter().fold(Value::new(0.0, Vec::new(), Vec::new()), |acc, l| Value::add(&acc, l)),
            &Value::new(n as f64, Vec::new(), Vec::new()),
        );

        // Backward pass
        loss.backward();

        // Adam optimizer update
        let lr_t = learning_rate * (1.0 - step as f64 / num_steps as f64);
        for (i, p) in params.iter().enumerate() {
            let grad = *p.grad.borrow();
            m[i] = beta1 * m[i] + (1.0 - beta1) * grad;
            v[i] = beta2 * v[i] + (1.0 - beta2) * grad * grad;
            let m_hat = m[i] / (1.0 - beta1.powi((step + 1) as i32));
            let v_hat = v[i] / (1.0 - beta2.powi((step + 1) as i32));
            
            // Update parameter data directly
            let new_data = p.data - lr_t * m_hat / (v_hat.sqrt() + eps_adam);
            unsafe {
                let p_mut = p.as_ref() as *const Value as *mut Value;
                (*p_mut).data = new_data;
            }
            p.zero_grad();
        }

        print!("\rstep {:4} / {:4} | loss {:.4}", step + 1, num_steps, loss.data);
        std::io::stdout().flush()?;
    }

    // Inference
    let temperature = 0.5;
    println!("\n--- inference (new, hallucinated names) ---");
    for sample_idx in 0..40 {
        let mut keys = vec![Vec::new(); n_layer];
        let mut values = vec![Vec::new(); n_layer];
        let mut token_id = bos;
        let mut sample = Vec::new();

        for pos_id in 0..block_size {
            let logits = gpt(
                token_id,
                pos_id,
                &mut keys,
                &mut values,
                &state_dict,
                n_layer,
                n_head,
                head_dim,
            );
            let probs = softmax(
                &logits
                    .iter()
                    .map(|l| Value::div(l, &Value::new(temperature, Vec::new(), Vec::new())))
                    .collect::<Vec<_>>(),
            );
            let weights: Vec<f64> = probs.iter().map(|p| p.data).collect();
            let dist = WeightedIndex::new(&weights).unwrap();
            token_id = dist.sample(&mut rng);
            if token_id == bos {
                break;
            }
            sample.push(uchars[token_id]);
        }

        println!("sample {:2}: {}", sample_idx + 1, sample.iter().collect::<String>());
    }

    Ok(())
}


