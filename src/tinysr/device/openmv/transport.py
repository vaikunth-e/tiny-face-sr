def millis():
    return time.ticks_ms()


def difference(start):
    return time.ticks_diff(millis(), start)


def memory():
    gc.collect()
    return {"free": int(gc.mem_free()), "allocated": int(gc.mem_alloc())}


def phase(name, **details):
    item = {"name": name, "memory": memory()}
    item.update(details)
    report["stages"].append(item)
    print(TAG + "|PHASE|" + name)
    time.sleep_ms(80)


def stream(kind, content, chunk_size, pause):
    # transfer one completed image or report with ordering and digest checks
    source = memoryview(content)
    total = len(source)
    if kind != "REPORT" and total > MAX_JPEG_BYTES:
        raise RuntimeError("encoded JPEG exceeds configured limit")
    digest = uhashlib.sha256()
    for start in range(0, total, 4096):
        digest.update(source[start : start + 4096])
    hexdigest = ubinascii.hexlify(digest.digest()).decode()
    chunks = (total + chunk_size - 1) // chunk_size
    print(TAG + "|START|%s|%d|%d|%s" % (kind, total, chunks, hexdigest))
    time.sleep_ms(250)
    for index in range(chunks):
        part = source[index * chunk_size : (index + 1) * chunk_size]
        encoded = ubinascii.b2a_base64(part).strip().decode()
        print(TAG + "|DATA|%s|%d|%s" % (kind, index, encoded))
        time.sleep_ms(pause)
    print(TAG + "|END|%s|%d" % (kind, chunks))
    time.sleep_ms(150)
