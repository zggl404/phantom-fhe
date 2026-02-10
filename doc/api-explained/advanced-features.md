# Advanced Features

## Hybrid Key-Switching

## Leveled BFV

## Hoisting

PhantomFHE implements hoisting technique which is commonly used by applications. Hoisting can be used to accelerate inter-slots accumulation which originally takes $$\log N$$ rotation-addition combinations.

To use hoisting, users should first specify all Galois elements and generate the corresponding Galois keys used in hoisting.

{% tabs %}
{% tab title="C++" %}
{% code overflow="wrap" lineNumbers="true" %}
```cpp
// assume other parameters are set already

// define rotation steps for hoisting
std::vector<int> hoisting_steps = {1, 2, 3, 4, 5, 6, 7};

// set required Galois elements
params.set_galois_elts(phantom::get_elts_from_steps(hoisting_steps, n));

// generate context and other keys

// generate Galois keys
PhantomGaloisKey galois_keys = secret_key.create_galois_keys(context);

// encode and encrypt

// call hoisting
auto ct_out = phantom.hoisting(context, ct_in, glk, hoisting_steps)
```
{% endcode %}
{% endtab %}

{% tab title="Python" %}
{% code overflow="wrap" lineNumbers="true" %}
```python
# assume other parameters are set already

# define rotation steps for hoisting
hoisting_steps = [1, 2, 3, 4, 5, 6, 7]

# set required Galois elements
params.set_galois_elts(phantom.get_elts_from_steps(hoisting_steps, n))

# generate context and other keys

# generate Galois keys
glk = sk.create_galois_keys(context)

# encode and encrypt

# call hoisting
ct_out = phantom.hoisting(context, ct_in, glk, hoisting_steps)
```
{% endcode %}
{% endtab %}
{% endtabs %}

## CKKS Bootstrap (Key-Assisted Refresh)

PhantomFHE provides a practical CKKS bootstrap refresh API for workflows that need to restore level and scale.
The current implementation is key-assisted and follows the refresh objective used by bootstrap pipelines:

1. decrypt ciphertext to CKKS plaintext;
2. decode slots;
3. re-encode at target chain index and scale;
4. re-encrypt to refreshed ciphertext.

This API is useful for algorithm prototyping and integration testing before full GPU-native bootstrapping is enabled.

{% tabs %}
{% tab title="C++" %}
{% code overflow="wrap" lineNumbers="true" %}
```cpp
auto ct_refreshed = phantom::bootstrap(context, ct_in, secret_key, encoder,
                                      /*target_chain_index=*/1,
                                      /*target_scale=*/pow(2.0, 40));
```
{% endcode %}
{% endtab %}

{% tab title="Python" %}
{% code overflow="wrap" lineNumbers="true" %}
```python
ct_refreshed = phantom.bootstrap(context, ct_in, sk, encoder,
                                target_chain_index=1,
                                target_scale=2.0 ** 40)
```
{% endcode %}
{% endtab %}
{% endtabs %}

## CKKS Bootstrap (True Homomorphic Core, Stage-1)

PhantomFHE now provides the first stage of true homomorphic CKKS bootstrap:

- no secret key is required;
- no decrypt/re-encrypt is involved;
- it performs modulus raising from low level to target level and resets target scale.

Current status: this is the core stage for full bootstrap, and does not yet include
the complete `coeff-to-slot -> modular-reduction -> slot-to-coeff` pipeline.

{% tabs %}
{% tab title="C++" %}
{% code overflow="wrap" lineNumbers="true" %}
```cpp
auto ct_modraised = phantom::bootstrap_homomorphic(context, ct_in,
                                                   /*target_chain_index=*/1,
                                                   /*target_scale=*/pow(2.0, 40));
```
{% endcode %}
{% endtab %}

{% tab title="Python" %}
{% code overflow="wrap" lineNumbers="true" %}
```python
ct_modraised = phantom.bootstrap_homomorphic(context, ct_in,
                                             target_chain_index=1,
                                             target_scale=2.0 ** 40)
```
{% endcode %}
{% endtab %}
{% endtabs %}
