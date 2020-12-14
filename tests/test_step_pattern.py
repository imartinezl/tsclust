#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tsclust.step_pattern


def test_step_pattern():
    for pattern_name in tsclust.step_pattern.__all__:
        sp = tsclust.step_pattern.get_pattern(pattern_name)
        print(sp.table())
